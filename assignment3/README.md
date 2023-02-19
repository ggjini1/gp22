# Assignment 3

Name: Gabriel Gjini

Legi-Nr: 15-932-437

## Required results
Edit this 'README.md' file to report all your results. You only need to update the tables in the reports section by adding screenshots and reporting results.

### Mandatory Tasks
1) Show the screenshots of mesh **torus.obj** and the mesh **Julius.obj** from the data folder shaded with the 5 different normals. For the PCA and quadratic fitted normals show them each with both k=1 and k=2. Note that here k means k-ring neighbors.

2) Report your matching between curvature expressions and the figures. Provide a short motivation for your matchings.

3) Show screenshots of mesh **bumpy-cube.obj** from the data folder coloured according to the 4 discrete curvature measures.

4) Report your findings on the differences between implicit and explicit Laplacian smoothing, the differences between uniform vs cotangent weights and report the parameters you used and how changing them affects the result. Also show screenshots of both the implicit and explicit Laplacian smoothing results for mesh **bunny_noise.obj** from the data folder.

5) Report your findings on the comparison with results you obtained with the Laplacian smoothing vs with bilateral smoothing. Show screenshots of bilateral smoothing applied to mesh **bunny_noise.obj** from the data folder.


## Reports
### 1 - Shading w.r.t. different normals

**Use the mesh torus.obj and the mesh Julius.obj**

**Use the built-in function igl::per_vertex_normals() to orient your normals consistently**

| normals        | torus.obj                  | Julius.obj                 |
| :-----------:  | ------------------------------------- |------------------------------------- |
| standard       |<img align="center" src="./res/torusstandard.png" width="300">| <img align="center"  src="./res/juliusstandard.png" width="300"> |
| area-weighted  |<img align="center" src="./res/torusarea.png" width="300">| <img align="center"  src="./res/juliusarea.png" width="300"> |
| mean-curvature |<img align="center" src="./res/torusmean.png" width="300">| <img align="center"  src="./res/juliusmean.png" width="300"> |
| PCA (k=1)      |<img align="center" src="./res/toruspca1.png" width="300">| <img align="center"  src="./res/juliuspca1.png" width="300"> |
| PCA (k=2)      |<img align="center" src="./res/toruspca2.png" width="300">| <img align="center"  src="./res/juliuspca2.png" width="300"> |
| quadratic (k=1)|<img align="center" src="./res/torusquad1.png" width="300">| <img align="center"  src="./res/juliusquad1.png" width="300"> |
| quadratic (k=2) |<img align="center" src="./res/torusquad2.png" width="300">| <img align="center"  src="./res/juliusquad2.png" width="300"> |

### 2 - Matching curvature expressions and the figures
| expression   |  Your answer |  Your explanation   |
|--------------|--------------|------------------|
| k1           | d      | This is the only plot where the curvature is never zero. This corresponds to expression k1.|
| k2           | c      | This is the only plot where the curvature changes sign. This hints toward the linear expression. |
| k3           | a      | By process of elimination. Also, the curvature nicely follows a quadratic, gently decreasing/increasing around 0.|
| k4           | b      | This is the only plot where the curvature changes sign twice. This corresponds to the curvature expression in k4.|


### 3 - Visualize curvatures

**Use the mesh bumpy-cube.obj**

| Min Curvature                         |  Max Curvature                       |
| ------------------------------------- |------------------------------------- |
|<img align="center" src="./res/cubemin.png" width="300">| <img align="center"  src="./res/cubemax.png" width="300"> |
| Mean Curvature                        |  Gaussian Curvature                  |
|<img align="center" src="./res/cubemean.png" width="300">| <img align="center"  src="./res/cubegauss.png" width="300"> |


### 4 - Implicit v.s. explicit Laplacian Smoothing

**Use the mesh bunny_noise.obj**

**Try different laplacian matrices, step sizes and iterations**

The parameter in the following are displayed as (N, D), where N denotes the number of iterations and D denotes the scaling parameter (following the notation of the lecture slides, this is dt*lambda).

| Input  |  Implicit (4, 0.1) uniform Laplacian|  Implicit (1, 0.001) cotangent Laplacian| Implicit (3, 0.001) cotangent Laplacian|
| -------|----------------------------- |------------------------------------|---------------------------------- |
|<img align="center" src="./res/bunnyinput.png" width="300">| <img align="center"  src="./res/impbunnyunif.png" width="300"> |<img align="center"  src="./res/impbunnycot.png" width="300"> |<img align="center"  src="./res/impbunnycotextreme.png" width="300"> |

Your observations:
Implicit smoothing seems to be very numerically robust. Given any parameter D, the method performs the smoothing after only very few iterations. The smaller the parameter D is, the more iterations we need to perform to get to the same smoothness of the surface. When we choose to use the cotangent Laplacian, this behaviour is even more appearant. In the third picture above we see that for a rather small value of D, we get a rather smooth surface after only one iteration. This leads to very extreme smoothing and degenerate surfaces when we perform too many iterations, see last picture above. We can counteract this by choosing a smaller value for D.

| Input  |  Explicit (100, 0.1) uniform Laplacian|  Explicit (150, 0.000001) cotangent Laplacian| Explicit (3, 0.00001) cotangent Laplacian      |
| -------|----------------------------- |------------------------------------|---------------------------------- |
|<img align="center" src="./res/bunnyinput.png" width="300">| <img align="center"  src="./res/expbunnyunif.png" width="300"> |<img align="center"  src="./res/expbunnycot.png" width="300"> |<img align="center"  src="./res/expbunnycotextreme.png" width="300"> |

Your observations:
The first thing that we can observe is that the explicit smoothing requires many more iterations that the implicit smoothing. When we choose the uniform Laplacian, we get a similar smoothing as with implicit smoothing with uniform Laplacian. Keeping the value of D fixed, we need to perform 10x the number of iterations to get similar results. When we use the cotangent Laplacian, however, we see some very numerically unstable behaviour. When we keep the value of D very small, we get reasonably smooth surfaces after many iterations, see picture 3 above. When we choose a value of D that is too large, we start to see some very spiky artifacts after only few iterations, see picture 4 above. When we perform more iterations at this point the surface degenerates and the spiky artifacts dominate the geometry.

### 5 - Laplacian v.s. bilateral smoothing

**Use the mesh bunny_noise.obj**

For bilateral smoothing, we choose standard normals and choose the neighborhood of a vertex to be its one-ring.

| Input                                 |  Laplacian Smoothing                 |  Bilateral Smoothing                 |
| ------------------------------------- |------------------------------------- |------------------------------------- |
|<img align="center" src="./res/bunnyinput.png" width="300">| <img align="center"  src="./res/bunnylap.png" width="300"> |<img align="center"  src="./res/bunnybil.png" width="300"> |

Your observations:
When compared to laplacian smoothing, the results of bilateral smoothing are more sensitive to the texture of the surface. In the example of the bunny, we see that with bilateral smoothing, the texture of the bunny's fur gets preserved. In contrast to that, Laplacian smoothin will simply treat the texture as noise and smooth it away. Another remarkable property of bilateral smoothing is that after only very few iterations, the method yields satisfying results. Any increase in the number of iterations after that will not improve the result at all. Similarly, the choice of the parameters sigma_s and sigma_c affect the result only very little. Both these phenomena can be explained by the normalization that is applied to the amount of drift towards the normal. However, if sigma_c is chosen too small, the neighborhood of queried vertices become very small or empty, leading to spiky artifacts and bigger holes.