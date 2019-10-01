# 2019-10-01
## Current progress/problem:
* Touch energy and prior energy explodes in opposite directions, cancelling out each other but makes training unstable
    * With 10x L2 regularization on both D and G, the energy no longer explodes but retracts instead to very small numbers. Possibly due to generator produce too small result each time. 
    * With 10x L2 on D and smaller L2 on G, the energy still explodes. 
    * **Proposed Solution** Try prior multiplier
    * **Proposed Solution** Try sigmoid energy