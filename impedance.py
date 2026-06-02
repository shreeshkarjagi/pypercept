"""sensing channel impedance from bipolar decomposition."""

import numpy as np
from dataclasses import dataclass

@dataclass
class ContactImpedance:
    """individual contact impedances and derived channel impedance for one hemisphere."""
    hemisphere: str
    ring_contact: str
    segment_contacts: list
    z_ring: float
    z_segments: dict          #{'sensight_2a': 1207, ...}
    z_parallel: float         #combined segment impedance
    channel_impedance: float  #z_ring + z_parallel
    raw_bipolars: dict        #the bipolar values used

def decompose_sensing_impedance(impedance_list, ring, segments):
    """
    bipolars-only decomposition, z_case never enters.
    impedance_list is session.impedance (already parsed from JSON).
    """
    #grab all bipolar pairs
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

    results = []
    for side in ["left", "right"]:
        sa, sb, sc = segments
        b_ab = bip.get((side, sa, sb))
        b_ac = bip.get((side, sa, sc))
        b_bc = bip.get((side, sb, sc))
        if None in (b_ab, b_ac, b_bc):
            continue
        z_a = (b_ab + b_ac - b_bc) / 2
        z_b = (b_ab + b_bc - b_ac) / 2
        z_c = (b_ac + b_bc - b_ab) / 2
        #ring from ring2seg bipolars
        z0s = []
        for seg, z_seg in [(sa, z_a), (sb, z_b), (sc, z_c)]:
            b_rs = bip.get((side, ring, seg))
            if b_rs is not None:
                z0s.append(b_rs - z_seg)
        z_ring = np.mean(z0s)
        z_par = 1 / (1/z_a + 1/z_b + 1/z_c)

        results.append(ContactImpedance(
            hemisphere=side,
            ring_contact=ring,
            segment_contacts=segments,
            z_ring=z_ring,
            z_segments={sa: z_a, sb: z_b, sc: z_c},
            z_parallel=z_par,
            channel_impedance=z_ring + z_par,
            raw_bipolars={k: v for k, v in bip.items() if k[0] == side},
        ))
    return results
