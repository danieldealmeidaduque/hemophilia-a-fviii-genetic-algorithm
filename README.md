# Project to diagnose hemophilia severity using only sequence of amino acids and a point mutation

## Description

This project have several functionalities such as:

Obs: Here "sequence" is the same as "sequence of amino acids"

### point_mutate_align_slice.py

#### input 
* wild sequence fasta file
* point mutation csv file

#### output
* align missense mutated sequences
* align sliced missense mutated sequences
* align nonsense mutated sequences
* align sliced nonsense mutated sequences

#### functionalities
* Point mutate a sequence
* Align multiple mutated sequences
* Slice aligned sequences 
 * Remove an amino acid if in a position have same amino acids for all the sequences

### colorize_svg_from_seaview.py

#### input
* black and white svg file

#### output
* colored svg file

#### functionalities
* Colorize a phylogenetic tree svg file from seaview
 * mild - green
 * moderate - blue
 * severe - red

### genetic_algorithm_hemophilia.py

#### input
* point mutation csv file
* amino acids distance matrix excel file
* relative surface area csv file

#### output

#### functionalities
* read files and combine then
  * link wild amino acid and new amino acid with distance matrix
  * link position hgvs with relative surface area
  * ... try to predict
  


