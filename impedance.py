"""assumes B_ij = z_i + z_j (additive contact model). this is
approximate. checks consistency via z_ring spread and flags
when the model doesn't fit.
"""

import re
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#max acceptable spread (ohms) between z_ring estimates from different segments
SPREAD_THRESHOLD = 200


@dataclass
class ChannelImpedance:
    """decomposed sensing channel impedance for one hemisphere."""
    hemisphere: str
    sensing_pair: tuple          #(level_a, level_b) e.g. (1, 3)
    channel_type: str            #ring-ring, ring-seg, seg-seg
    channel_impedance: float
    contacts: dict               #per-level breakdown
    model_valid: bool
    validation_notes: list


def _parse_bipolar_table(impedance_list):
    """flat lookup of all bipolar pairs."""
    bip = {}
    for rec in impedance_list or []:
        for hemi in rec.get("Hemisphere", []):
            side = "left" if "Left" in hemi.get("Hemisphere", "") else "right"
            for bp in hemi.get("SessionImpedance", {}).get("Bipolar", []):
                e1 = bp["Electrode1"].replace("ElectrodeDef.", "").lower()
                e2 = bp["Electrode2"].replace("ElectrodeDef.", "").lower()
                v = bp.get("ResultValue")
                if v is not None:
                    bip[(side, e1, e2)] = float(v)
                    bip[(side, e2, e1)] = float(v)
    return bip


def _detect_levels(bip):
    """auto-detect electrode levels and whether they are segmented."""
    electrodes = set()
    for (side, e1, e2) in bip:
        electrodes.add(e1)
        electrodes.add(e2)

    levels = {}
    is_sensight = any("sensight" in e for e in electrodes)

    if is_sensight:
        for e in electrodes:
            m = re.match(r"sensight_(\d)([abc])?", e)
            if not m:
                continue
            lvl = int(m.group(1))
            if lvl not in levels:
                levels[lvl] = {"electrodes": [], "segmented": False}
            levels[lvl]["electrodes"].append(e)
            if m.group(2):
                levels[lvl]["segmented"] = True
    else:
        #1x4 lead: all rings, no segments
        #electrode naming not verified on real 1x4 data
        for e in electrodes:
            m = re.match(r"e(\d+)", e)
            if m:
                lvl = int(m.group(1))
                levels[lvl] = {"electrodes": [e], "segmented": False}

    return levels, is_sensight


def _decompose_segments(bip, side, seg_electrodes):
    """3 equations 3 unknowns from seg2seg bipolars."""
    sa, sb, sc = sorted(seg_electrodes)
    b_ab = bip.get((side, sa, sb))
    b_ac = bip.get((side, sa, sc))
    b_bc = bip.get((side, sb, sc))
    if None in (b_ab, b_ac, b_bc):
        return None
    return {
        sa: (b_ab + b_ac - b_bc) / 2,
        sb: (b_ab + b_bc - b_ac) / 2,
        sc: (b_ac + b_bc - b_ab) / 2,
    }


def _parallel(impedances):
    return 1 / sum(1/z for z in impedances)


def _detect_sensing_config(groups):
    """pull sensing config from Groups.Final (or Initial)."""
    configs = {}
    word_to_num = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3}
    for state in ["Final", "Initial"]:
        for g in groups.get(state, []):
            for sc in g.get("ProgramSettings", {}).get("SensingChannel", []):
                hemi = sc.get("HemisphereLocation", "")
                side = "left" if "Left" in hemi else "right"
                ch = sc.get("Channel", "").replace("SensingElectrodeConfigDef.", "").upper()
                parts = ch.replace("_AND_", "_").split("_")
                nums = [word_to_num[p] for p in parts if p in word_to_num]
                if len(nums) == 2 and side not in configs:
                    configs[side] = tuple(nums)
    return configs


def sensing_channel_impedance(session=None, impedance_list=None, groups=None, sensing_pair=None):
    """
    compute sensing channel impedance from bipolar decomposition.

    accepts either a PerceptSession or raw impedance_list + groups.
    auto-detects lead type and config, or accepts sensing_pair override.

    three cases:
      ring + ring:     channel = bipolar directly (exact)
      ring + segments: channel = z_ring + parallel(z_a, z_b, z_c)
      seg + seg:       channel = parallel(level_a) + parallel(level_b)
    """
    if session is not None:
        impedance_list = session.impedance
        if groups is None:
            groups = session.groups

    bip = _parse_bipolar_table(impedance_list)
    if not bip:
        logger.warning("no bipolar impedance data found")
        return {}

    levels, is_sensight = _detect_levels(bip)
    logger.info(f"detected {'SenSight' if is_sensight else '1x4'} lead, "
                f"levels: {sorted(levels.keys())}")

    if sensing_pair is not None:
        configs = {"left": sensing_pair, "right": sensing_pair}
    elif groups is not None:
        configs = _detect_sensing_config(groups)
    else:
        logger.warning("need either sensing_pair or groups to determine config")
        return {}

    results = {}
    for side in ["left", "right"]:
        if side not in configs:
            continue
        lev_a, lev_b = configs[side]
        if lev_a not in levels or lev_b not in levels:
            continue

        info_a, info_b = levels[lev_a], levels[lev_b]
        notes = []

        #ring-ring: bipolar is the channel directly
        if not info_a["segmented"] and not info_b["segmented"]:
            e_a, e_b = info_a["electrodes"][0], info_b["electrodes"][0]
            ch = bip.get((side, e_a, e_b))
            if ch is None:
                continue
            results[side] = ChannelImpedance(
                hemisphere=side, sensing_pair=(lev_a, lev_b),
                channel_type="ring-ring", channel_impedance=ch,
                contacts={lev_a: {"electrode": e_a},
                          lev_b: {"electrode": e_b}},
                model_valid=True, validation_notes=["exact"])
            continue

        #at least one side segmented
        contact_info = {}
        valid = True

        for lev, info in [(lev_a, info_a), (lev_b, info_b)]:
            if info["segmented"]:
                decomp = _decompose_segments(bip, side, info["electrodes"])
                if decomp is None:
                    notes.append(f"level {lev}: missing seg2seg bipolars")
                    valid = False
                    break
                contact_info[lev] = {
                    "z_segments": decomp,
                    "z_combined": _parallel(decomp.values()),
                }
            else:
                contact_info[lev] = {
                    "electrode": info["electrodes"][0],
                    "z_ring": None,
                }

        if not valid or len(contact_info) < 2:
            continue

        #solve z_ring for ring contacts from ring2seg bipolars
        for lev, cinfo in contact_info.items():
            if "z_ring" not in cinfo:
                continue
            other_lev = [l for l in contact_info if l != lev][0]
            other_segs = contact_info[other_lev]["z_segments"]
            e_ring = cinfo["electrode"]
            z_ring_estimates = []
            for e_seg, z_seg in other_segs.items():
                b_rs = bip.get((side, e_ring, e_seg))
                if b_rs is not None:
                    z_ring_estimates.append(b_rs - z_seg)
            if not z_ring_estimates:
                notes.append(f"level {lev}: no ring2seg bipolars")
                valid = False
                continue
            cinfo["z_ring"] = np.mean(z_ring_estimates)
            spread = max(z_ring_estimates) - min(z_ring_estimates)
            cinfo["z_ring_spread"] = spread
            cinfo["z_ring_estimates"] = z_ring_estimates

            if cinfo["z_ring"] < 0:
                notes.append(f"level {lev}: z_ring={cinfo['z_ring']:.0f} (negative)")
                valid = False
            if spread > SPREAD_THRESHOLD:
                notes.append(f"level {lev}: z_ring spread={spread:.0f} > {SPREAD_THRESHOLD}")
                valid = False

        #channel impedance
        z_parts = []
        for lev in [lev_a, lev_b]:
            ci = contact_info[lev]
            if "z_segments" in ci:
                z_parts.append(ci["z_combined"])
            elif ci.get("z_ring") is not None:
                z_parts.append(ci["z_ring"])
            else:
                valid = False

        if len(z_parts) == 2:
            seg_count = sum(1 for i in [info_a, info_b] if i["segmented"])
            ctype = ["ring-ring", "ring-seg", "seg-seg"][seg_count]
            results[side] = ChannelImpedance(
                hemisphere=side, sensing_pair=(lev_a, lev_b),
                channel_type=ctype,
                channel_impedance=sum(z_parts),
                contacts=contact_info,
                model_valid=valid,
                validation_notes=notes)

    return results
