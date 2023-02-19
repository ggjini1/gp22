# Assignment 4

Name: Gabriel Gjini

Legi-Nr: 15-932-437

## Required results
Edit this 'README.md' file to report all your results. You only need to update the tables in the reports section by adding screenshots and reporting results.

### Mandatory Tasks

1) Screenshots of the parameterizations and textured (checkerboard) models for all the implemented methods and boundary conditions (models: cathead.obj, hemisphere.off, hemisphere_non_convex_boundary.off, Octo_cut2.obj)

2) Several examples of the distortion visualizations.


## Reports
### (mandatory-1) parameterization and checkerboard texture models
#### cathead
| Method            | checkerboard textured models          |         Parameterization             |
| :--------------:  | ------------------------------------- |------------------------------------- |
| Uniform (fixed)   |<img align="center" src="./res/catheadu.png" width="450">| <img align="center"  src="./res/paramcatheadu.png" width="450"> |
| Cotangent (fixed) |<img align="center" src="./res/catheadc.png" width="450">| <img align="center"  src="./res/paramcatheadc.png" width="450"> |
| LSCM (fixed)      |<img align="center" src="./res/catheadl.png" width="450">| <img align="center"  src="./res/paramcatheadl.png" width="450"> |
| ARAP (fixed)      |<img align="center" src="./res/catheada.png" width="450">| <img align="center"  src="./res/paramcatheada.png" width="450"> |
| LSCM (free)       |<img align="center" src="./res/catheadlfree.png" width="450">| <img align="center"  src="./res/paramcatheadlfree.png" width="450"> |
| ARAP (free)       |<img align="center" src="./res/catheadafree.png" width="450">| <img align="center"  src="./res/paramcatheadafree.png" width="450"> |

#### hemisphere
| Method            | checkerboard textured models          |         Parameterization             |
| :--------------:  | ------------------------------------- |------------------------------------- |
| Uniform (fixed)   |<img align="center" src="./res/hemu.png" width="450">| <img align="center"  src="./res/paramhemu.png" width="450"> |
| Cotangent (fixed) |<img align="center" src="./res/hemc.png" width="450">| <img align="center"  src="./res/paramhemc.png" width="450"> |
| LSCM (fixed)      |<img align="center" src="./res/heml.png" width="450">| <img align="center"  src="./res/paramheml.png" width="450"> |
| ARAP (fixed)      |<img align="center" src="./res/hema.png" width="450">| <img align="center"  src="./res/paramhema.png" width="450"> |
| LSCM (free)       |<img align="center" src="./res/hemlfree.png" width="450">| <img align="center"  src="./res/paramhemlfree.png" width="450"> |
| ARAP (free)       |<img align="center" src="./res/hemafree.png" width="450">| <img align="center"  src="./res/paramhemafree.png" width="450"> |


#### hemisphere_non_convex_boundary
| Method            | checkerboard textured models          |         Parameterization             |
| :--------------:  | ------------------------------------- |------------------------------------- |
| Uniform (fixed)   |<img align="center" src="./res/hemncu.png" width="450">| <img align="center"  src="./res/paramhemncu.png" width="450"> |
| Cotangent (fixed) |<img align="center" src="./res/hemncc.png" width="450">| <img align="center"  src="./res/paramhemncc.png" width="450"> |
| LSCM (fixed)      |<img align="center" src="./res/hemncl.png" width="450">| <img align="center"  src="./res/paramhemncl.png" width="450"> |
| ARAP (fixed)      |<img align="center" src="./res/hemnca.png" width="450">| <img align="center"  src="./res/paramhemnca.png" width="450"> |
| LSCM (free)       |<img align="center" src="./res/hemnclfree.png" width="450">| <img align="center"  src="./res/paramhemnclfree.png" width="450"> |
| ARAP (free)       |<img align="center" src="./res/hemncafree.png" width="450">| <img align="center"  src="./res/paramhemncafree.png" width="450"> |


#### Octo_cut2
| Method            | checkerboard textured models          |         Parameterization             |
| :--------------:  | ------------------------------------- |------------------------------------- |
| Uniform (fixed)   |<img align="center" src="./res/octu.png" width="450">| <img align="center"  src="./res/paramoctu.png" width="450"> |
| Cotangent (fixed) |<img align="center" src="./res/octc.png" width="450">| <img align="center"  src="./res/paramoctc.png" width="450"> |
| LSCM (fixed)      |<img align="center" src="./res/octl.png" width="450">| <img align="center"  src="./res/paramoctl.png" width="450"> |
| ARAP (fixed)      |<img align="center" src="./res/octa.png" width="450">| <img align="center"  src="./res/paramocta.png" width="450"> |
| LSCM (free)       |<img align="center" src="./res/octlfree.png" width="450">| <img align="center"  src="./res/paramoctlfree.png" width="450"> |
| ARAP (free)       |<img align="center" src="./res/octafree.png" width="450">| <img align="center"  src="./res/paramoctafree.png" width="450"> |



### (mandatory-2) distortion visualization
#### cathead
| mtd \ metric      | Conformal (angle) |    Authalic (area)  |  Isometric  (length)    |
| :--------------:  | ----------------- | ------------------- | ----------------------- |
| LSCM (free)       |<img align="center" src="./res/lcatconf.png" width="300">| <img align="center"  src="./res/lcataut.png" width="300"> | <img align="center"  src="./res/lcatiso.png" width="300"> |
| ARAP (free)       |<img align="center" src="./res/acatconf.png" width="300">| <img align="center"  src="./res/acataut.png" width="300"> |<img align="center"  src="./res/acatiso.png" width="300"> |


#### hemisphere
| mtd \ metric      | Conformal (angle) |    Authalic (area)  |  Isometric  (length)    |
| :--------------:  | ----------------- | ------------------- | ----------------------- |
| LSCM (free)       |<img align="center" src="./res/lhemconf.png" width="300">| <img align="center"  src="./res/lhemaut.png" width="300"> | <img align="center"  src="./res/lhemiso.png" width="300"> |
| ARAP (free)       |<img align="center" src="./res/ahemconf.png" width="300">| <img align="center"  src="./res/ahemaut.png" width="300"> |<img align="center"  src="./res/ahemiso.png" width="300"> |
