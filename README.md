This computational workflow was developed to map the rotational energy landscape of the bacterial flaggelum (PDB: 7CG0), but can be used and extended to any Rosetta representable (peptides famously, buy also any molecules) natural** or synthetic 'nanomachine' assembly. 


  In this example, the structure of the bacterial flagellum was retrieved from the protein data bank (PDB: 7CG0, `7CGO_full1.png`), aligned in xy to the cyclic symmetry axis, and truncated to only account for the molecular interactions at the interface between the stator and rotor (`7CGO_trunc*png`). 
  
  The rotational landscape was generated by rotating stator and rotor components relative to each other along the symmetry axis and sampling with a 0.1 degree angle (`dock_rotor_axle_7CGO.py`), producing a new structure for each conformation (`/example_sampled_rotations/*pdb`. Rosetta energies (`Total_score`) were then computed for 3600 rotational bins x 10 trajectories for the whole stator plus rotor complex after repacking (`PackRotamersMover`) all residues (`repack.xml`). Inspection of the protein-protein interface of rotamer minizimed structures between stator/rotor reveals low energy configurations and intricate hydrogen bonded interactions (`7CGO_interfcaehbind1.png`). 

  The rotational energy landscape can be efficiently visualized as a polar plot showing the mean energies (+/-s.d.) (`total_score_7CGO_centered.png`), with symmetric 'spikes' corresponding to high energies. A FFT representation of the computed energy landscape reveals higher amplitude frequencies corresponding to expected "stepping" behavior (`7CGO_plot.ipynb`)



**To be published with: Rieu et al., `Single-molecule observation of multiscale dynamics in the molecular bearing of the bacterial flagellum`
