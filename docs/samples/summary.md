
### TOPOLOGICAL MATCHING PERFORMANCE RESULT

| Test Case Name | GT Nodes | Total Detected | Misclassified | False detection | TP | Misalignments (MA) | FP | Skip | Precision | Recall | Avg Dist (pixels) |Max Dist (pixels) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| clean_baseline_0001 | 961 | 961 | 2 | 0 | 961 | 0 | 0 | 0 | 100.000 | 100.000 | 0.7 | 1.6 |
| oblique_tilt_high_0004 | 952 | 889 | 63 | 0 | 888 | 0 | 0 | 1 | 100.000 | 93.277 | 0.7 | 1.6 |
| roll_120_0010 | 924 | 890 | 8 | 8 | 880 | 0 | 0 | 2 | 100.000 | 95.238 | 0.7 | 1.4 |
| roll_180_0013 | 961 | 976 | 14 | 17 | 958 | 0 | 0 | 1 | 100.000 | 99.688 | 0.8 | 1.6 |
| roll_240_0016 | 922 | 636 | 60 | 10 | 561 | 0 | 0 | 65 | 100.000 | 60.846 | 0.8 | 1.7 |
| roll_300_0019 | 924 | 887 | 11 | 9 | 877 | 0 | 0 | 1 | 100.000 | 94.913 | 0.8 | 1.6 |
| roll_60_0007 | 922 | 900 | 12 | 18 | 882 | 0 | 0 | 0 | 100.000 | 95.662 | 0.8 | 1.7 |

***
### Definitions


***
* **$\text{Misclassified (Corrected)}$:** Tracks the number of physical nodes whose binary shape labels (circles vs. triangles) were initially misidentified by the low-level detector, but were **successfully resolved and corrected** by the decoder. They are fully included in the $TP$ pool.
* **$\text{False Detection (Ghost)}$:** Pure background raster noise or glare reflections erroneously picked up by the sensor but blocked from entering the lattice by the topological filter. They are discarded from further processing.
* **$\text{Skip}$ (Isolated Nodes):** Valid physical grid features that were successfully detected on the sensor but left with a graph adjacency degree of zero because surrounding marker dropouts prevented them from anchoring to a major topological island. They are discarded from further processing.
* All evaluation ratio bounds follow standard photogrammetric confusion matrix principles:
  $$\text{Precision} = \frac{TP}{TP + FP + MA}$$
  $$\text{Recall} = \frac{TP}{\text{Total Visible GT Nodes}}$$

***
*Generated automatically by Hexagonal Galois Pattern Matching Engine*
