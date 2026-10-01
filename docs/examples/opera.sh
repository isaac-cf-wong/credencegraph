#!/usr/bin/env bash
# The argument of the 2011 OPERA neutrino-velocity measurement, encoded by hand.
# Bases and strengths are one reader's judgement as of late 2011, not the paper's.
set -euo pipefail

credencegraph init opera.json

# The measurement and the premises it rests on.
credencegraph add-node opera.json --id timing --base beta:9,1 --source arXiv:1109.4897 \
    --statement "The timing chain, GPS synchronisation, fibre delays and electronics, is calibrated within the quoted systematic uncertainty."
credencegraph add-node opera.json --id baseline --base beta:99,1 --source arXiv:1109.4897 \
    --statement "The 730 km baseline between CERN and Gran Sasso is known to about 20 cm."
credencegraph add-node opera.json --id extraction --base beta:9,1 --source arXiv:1109.4897 \
    --statement "Fitting the proton waveform to the neutrino arrival times gives an unbiased delay."
credencegraph add-node opera.json --id bunched --base 0.95 --source arXiv:1109.4897v2 \
    --statement "A rerun with 3 ns proton bunches, timing each event directly, finds a consistent early arrival."
credencegraph add-node opera.json --id early --base 0.999 --stated 0.999999998 --source arXiv:1109.4897 \
    --statement "The neutrinos arrive 60.7 ± 6.9 (stat) ± 7.4 (sys) ns earlier than light would, a 6.0 sigma effect."
credencegraph relate opera.json timing early --type requires --strength 0.95
credencegraph relate opera.json baseline early --type requires --strength 0.95
credencegraph relate opera.json extraction early --type requires --strength 0.9
credencegraph relate opera.json bunched extraction --type supports --strength 0.8

# The claim, the measurement as a reason for it, and the reasons against it.
credencegraph add-node opera.json --id faster --base 0.001 --source arXiv:1109.4897 \
    --statement "Muon neutrinos of about 17 GeV travel faster than light, with (v - c)/c of about 2.5e-5."
credencegraph add-node opera.json --id sn1987a --base 0.99 --source doi:10.1103/PhysRevD.36.3276 \
    --statement "Neutrinos from SN1987A arrived within hours of its light after 168,000 years, so |v - c|/c is below about 2e-9 at 10 MeV."
credencegraph add-node opera.json --id pair-emission --base 0.9 --source arXiv:1109.6562 \
    --statement "Neutrinos this superluminal would shed energy by emitting electron-positron pairs, depleting the beam's high-energy part before it reached Gran Sasso."
credencegraph relate opera.json early faster --type supports --strength 0.95
credencegraph relate opera.json sn1987a faster --type refutes --strength beta:7,3
credencegraph relate opera.json pair-emission faster --type refutes --strength beta:8,2

credencegraph check opera.json
credencegraph query opera.json marginal faster
credencegraph diagnose opera.json --target faster
