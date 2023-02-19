#include <iostream>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/jet.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/sum.h>
#include <igl/speye.h>
#include <igl/bfs.h>
#include <igl/cotmatrix.h>
#include <igl/principal_curvature.h>
#include <imgui/imgui.h>
/*** insert any libigl headers here ***/
#include <igl/doublearea.h>
#include <igl/massmatrix.h>


using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Vertex array, #Vx3
Eigen::MatrixXd V;
// Face array, #Fx3
Eigen::MatrixXi F;
//Face normals #Fx3
Eigen::MatrixXd FN;
//Vertex normals #Vx3
Eigen::MatrixXd VN;

// Per-vertex uniform normal array, #Vx3
Eigen::MatrixXd N_uniform;
// Per-vertex area-weighted normal array, #Vx3
Eigen::MatrixXd N_area;
// Per-vertex mean-curvature normal array, #Vx3
Eigen::MatrixXd N_meanCurvature;
// Per-vertex PCA normal array, #Vx3
Eigen::MatrixXd N_PCA;
// Per-vertex quadratic fitted normal array, #Vx3
Eigen::MatrixXd N_quadraticFit;

// Per-vertex mean curvature, #Vx3
Eigen::VectorXd K_mean;
// Per-vertex Gaussian curvature, #Vx3
Eigen::VectorXd K_Gaussian;
// Per-vertex minimal principal curvature, #Vx3
Eigen::VectorXd K_min_principal;
// Per-vertex maximal principal curvature, #Vx3
Eigen::VectorXd K_max_principal;
// Per-vertex color array, #Vx3
Eigen::MatrixXd colors_per_vertex;

// Explicitely smoothed vertex array, #Vx3
Eigen::MatrixXd V_expLap;
// Implicitely smoothed vertex array, #Vx3
Eigen::MatrixXd V_impLap;
// Bilateral smoothed vertex array, #Vx3
Eigen::MatrixXd V_bilateral;

int k = 1;
int Num_iter = 10;
double Delta = 0.01;
int takeUniformL = 1;
double sigmac = 0.1;
double sigmas = 0.1;

void getKRingNeighborMatrix(
    const std::vector<std::vector<int>>& vertexAdjacency, 
    const int vertexIndex, 
    const int k, 
    Eigen::MatrixXd& neighborMatrix)
{   
    // Collect indices of neighboring vertices
    std::vector<int> kRingNeighbors;
    kRingNeighbors.push_back(vertexIndex);
    kRingNeighbors = vertexAdjacency[vertexIndex];
    if(k == 2)
    {
        int numberOfNeighbors = kRingNeighbors.size();
        for(int i = 0; i < numberOfNeighbors; ++i)
        {
            kRingNeighbors.insert(
                std::end(kRingNeighbors),
                std::begin(vertexAdjacency[i]),
                std::end(vertexAdjacency[i]));
        }
        std::sort(kRingNeighbors.begin(), kRingNeighbors.end());
        auto iter = std::unique(kRingNeighbors.begin(), kRingNeighbors.end());
        kRingNeighbors.erase(iter, kRingNeighbors.end());
    }
    // Create Matrix from neighbor Inidces
    neighborMatrix.resize(kRingNeighbors.size(), 3);
    for(int i = 0; i < kRingNeighbors.size(); ++i)
    {
        neighborMatrix.row(i) = V.row(kRingNeighbors[i]);
    }
}

void getKRingNeighborMatrixWithinSigmaC(
    const std::vector<std::vector<int>>& vertexAdjacency, 
    const int vertexIndex, 
    const int k, 
    Eigen::MatrixXd& neighborMatrix)
{   
    // Collect indices of neighboring vertices
    std::vector<int> kRingNeighbors;
    kRingNeighbors.push_back(vertexIndex);
    kRingNeighbors = vertexAdjacency[vertexIndex];
    if(k == 2)
    {
        int numberOfNeighbors = kRingNeighbors.size();
        for(int i = 0; i < numberOfNeighbors; ++i)
        {
            kRingNeighbors.insert(
                std::end(kRingNeighbors),
                std::begin(vertexAdjacency[i]),
                std::end(vertexAdjacency[i]));
        }
        std::sort(kRingNeighbors.begin(), kRingNeighbors.end());
        auto iter = std::unique(kRingNeighbors.begin(), kRingNeighbors.end());
        kRingNeighbors.erase(iter, kRingNeighbors.end());
    }
    // Create Matrix from neighbor Inidces
    neighborMatrix.resize(kRingNeighbors.size(), 3);
    for(int i = 0; i < kRingNeighbors.size(); ++i)
    {
        double distance = (V.row(kRingNeighbors[i]) - V.row(vertexIndex)).squaredNorm();
        if(distance <= 2*sigmac*sigmac)
        {
            neighborMatrix.row(i) = V.row(kRingNeighbors[i]);
        }
    }
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing uniform vertex normals here:
        // store in N_uniform
        igl::per_face_normals(V,F,FN);
        N_uniform.setZero(V.rows(),3);
        for(int i = 0;i<F.rows();i++)
        {
          for(int j = 0; j < 3;j++)
          {
            N_uniform.row(F(i,j)) += FN.row(i);
          }
        }
        // Use igl::per_vertex_normals to orient your normals consistently.
        // Set the viewer normals.
        N_uniform.rowwise().normalize();
        viewer.data().set_normals(N_uniform);
    }

    if (key == '2') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing area-weighted vertex normals here:
        Eigen::VectorXd FA;
        igl::doublearea(V,F,FA);
        N_area.setZero(V.rows(),3);
        for(int i = 0;i<F.rows();i++)
        {
          // throw normal at each corner
          for(int j = 0; j < 3;j++)
          {
            N_area.row(F(i,j)) += FA(i) * FN.row(i);
          }
        }
        // Use igl::per_vertex_normals to orient your normals consistently.
        // Set the viewer normals.
        N_area.rowwise().normalize();
        viewer.data().set_normals(N_area);
    }

    if (key == '3') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing mean-curvature vertex normals here:
        // store in N_meanCurvature
        Eigen::SparseMatrix<double> L,M,Minv;
        igl::cotmatrix(V,F,L);
        igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
        igl::invert_diag(M,Minv);
        N_meanCurvature = -Minv*(L*V);
        // Use igl::per_vertex_normals to orient your normals consistently.
        Eigen::MatrixXd Ntmp;
        igl::per_vertex_normals(V,F,Ntmp);
        for(int i = 0;i<N_meanCurvature.rows();i++)
        {
          if(N_meanCurvature.row(i).dot(Ntmp.row(i)) < 0)
          {
            N_meanCurvature.row(i) *= -1;
          }
        }
        // Set the viewer normals.
        N_meanCurvature.rowwise().normalize();
        viewer.data().set_normals(N_meanCurvature);
    }

    if (key == '4') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing PCA vertex normals here:
        // store in N_PCA
        N_PCA.resize(V.rows(), 3);
        std::vector<std::vector<int>> vertexAdjacency;
        igl::adjacency_list(F, vertexAdjacency);
        for(int i = 0; i < V.rows(); ++i)
        {
            Eigen::MatrixXd neighborMatrix;
            // Compute Vertex-vertex Adjacency
            getKRingNeighborMatrix(vertexAdjacency, i, k, neighborMatrix);
            // Perform PCA
            Eigen::MatrixXd centered = neighborMatrix.rowwise() - neighborMatrix.colwise().mean();
            Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(neighborMatrix.rows() - 1);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(cov);
	        Eigen::MatrixXd eigenVectors = eigen.eigenvectors();
            Eigen::VectorXd eigenValues = eigen.eigenvalues();
            // Assign smallest Eigenvector
            N_PCA.row(i) << eigenVectors.col(0).transpose();

        }
        // Use igl::per_vertex_normals to orient your normals consistently.
        Eigen::MatrixXd Ntmp;
        igl::per_vertex_normals(V,F,Ntmp);
        for(int i = 0;i<N_PCA.rows();i++)
        {
          if(N_PCA.row(i).dot(Ntmp.row(i)) < 0)
          {
            N_PCA.row(i) *= -1;
          }
        }
        // Set the viewer normals.
        //N_PCA.rowwise().normalize();
        viewer.data().set_normals(N_PCA);
    }

    if (key == '5') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing quadratic fitted vertex normals here:
        // store in N_quadraticFit
        N_quadraticFit.resize(V.rows(), 3);
        std::vector<std::vector<int>> vertexAdjacency;
        igl::adjacency_list(F, vertexAdjacency);
        for(int i = 0; i < V.rows(); ++i)
        {
            Eigen::MatrixXd neighborMatrix;
            // Compute Vertex-vertex Adjacency
            getKRingNeighborMatrix(vertexAdjacency, i, k, neighborMatrix);
            // Perform PCA
            Eigen::MatrixXd centered = neighborMatrix.rowwise() - neighborMatrix.colwise().mean();
            Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(neighborMatrix.rows() - 1);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(cov);
	        Eigen::MatrixXd eigenVectors = eigen.eigenvectors();
            Eigen::VectorXd principalComp = eigenVectors.col(0);
            // First rotation matrix
            Eigen::VectorXd xyPC(3); xyPC << principalComp(0),principalComp(1),0;
            Eigen::VectorXd basisy(3); basisy << 0,1,0;
            double alpha = basisy.dot(xyPC);
            Eigen::MatrixXd rotMatrix1(3,3);
            rotMatrix1 << std::cos(alpha), -std::sin(alpha), 0,
                         std::sin(alpha), std::cos(alpha), 0,
                         0, 0, 1;
            // Second rotation matrix
            Eigen::VectorXd rotatedPC = rotMatrix1 * xyPC;
            Eigen::VectorXd basisz(3); basisz << 0,0,1;
            double beta = basisz.dot(rotatedPC);
            Eigen::MatrixXd rotMatrix2(3,3);
            rotMatrix2 << 1,0,0,
                          0, std::cos(beta), -std::sin(beta),
                          0, std::sin(beta), std::cos(beta);
            // Combine rotation matrices
            Eigen::MatrixXd rotMatrix = rotMatrix2 * rotMatrix1;
            // Fit bivariate polynomial
            Eigen::MatrixXd A, rotatedNeighborMatrix;
            Eigen::VectorXd B;
            rotatedNeighborMatrix.resize(neighborMatrix.rows(), 3);
            for(int j = 0; j < neighborMatrix.rows(); ++j)
            {
                rotatedNeighborMatrix.row(j) = rotMatrix * neighborMatrix.row(j);
            }
            A.resize(rotatedNeighborMatrix.rows(), 6);
            B.resize(rotatedNeighborMatrix.rows());
            for(int j = 0; j < rotatedNeighborMatrix.rows(); ++j)
            {
                double px, py, pz;
                px = rotatedNeighborMatrix(j,0); py = rotatedNeighborMatrix(j,1); pz = rotatedNeighborMatrix(j,2);
                A << px*px, py*py, px*py, px, py, 1;
                B << pz;
            }
            Eigen::VectorXd c = A.colPivHouseholderQr().solve(B);
            // Compute the partial derivatives and set them as directions for normal
            Eigen::VectorXd rotatedCurrentPoint;
            rotatedCurrentPoint = rotMatrix * V.row(i);
            double px, py, pz;
            px = rotatedCurrentPoint(0); py = rotatedCurrentPoint(1); pz = rotatedCurrentPoint(2);
            double dpdx = 2*c(0)*px + c(1)*py*py + c(2)*py + c(3) + c(4)*py;
            double dpdy = c(0)*px*px + 2*c(1)*py + c(2)*px + c(3)*px + c(4);
            Eigen::VectorXd rotatedNormal(3); rotatedNormal << dpdx, dpdy, 1;
            Eigen::MatrixXd rotMatrixInv = rotMatrix.inverse();
            N_quadraticFit.row(i) << rotMatrixInv * rotatedNormal;
        }
        // Use igl::per_vertex_normals to orient your normals consistently.
        Eigen::MatrixXd Ntmp;
        igl::per_vertex_normals(V,F,Ntmp);
        for(int i = 0;i<N_quadraticFit.rows();i++)
        {
          if(N_quadraticFit.row(i).dot(Ntmp.row(i)) < 0)
          {
            N_quadraticFit.row(i) *= -1;
          }
        }
        // Set the viewer normals.
        N_quadraticFit.rowwise().normalize();
        viewer.data().set_normals(N_quadraticFit);
    }

    if (key == '6') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        colors_per_vertex.setZero(V.rows(),3);
        // Add your code for computing the discrete mean curvature:
        // store in K_mean
        // Alternative discrete mean curvature
        Eigen::MatrixXd PD1,PD2;
        Eigen::VectorXd PV1,PV2;
        igl::principal_curvature(V,F,PD1,PD2,PV1,PV2);
        // mean curvature
        K_mean = 0.5*(PV1+PV2);
        // For visualization, better to normalize the range of K_mean with the maximal and minimal curvatures.
        // store colors in colors_per_vertex
        igl::jet(K_mean, K_mean.minCoeff(), K_mean.maxCoeff(), colors_per_vertex);
        // Set the viewer colors
        viewer.data().set_colors(colors_per_vertex);
    }

    if (key == '7') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        colors_per_vertex.setZero(V.rows(),3);
        // Add your code for computing the discrete Gaussian curvature:
        // store in K_Gaussian
        igl::gaussian_curvature(V,F,K_Gaussian);
        // For visualization, better to normalize the range of K_Gaussian with the maximal and minimal curvatures.
        // store colors in colors_per_vertex
        igl::jet(K_Gaussian, K_Gaussian.minCoeff(), K_Gaussian.maxCoeff(), colors_per_vertex);
        // Set the viewer colors
        viewer.data().set_colors(colors_per_vertex);
    }

    if (key == '8') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        colors_per_vertex.setZero(V.rows(),3);
        // Add your code for computing the discrete minimal principal curvature:
        // store in K_min_principal
        Eigen::MatrixXd PD1,PD2;
        Eigen::VectorXd PV1,K_min_principal;
        igl::principal_curvature(V,F,PD1,PD2,PV1,K_min_principal);
        // For visualization, better to normalize the range of K_min_principal with the maximal and minimal curvatures.
        // store colors in colors_per_vertex
        igl::jet(K_min_principal, K_min_principal.minCoeff(), K_min_principal.maxCoeff(), colors_per_vertex);
        // Uncomment the code below to draw a blue segment parallel to the minimal curvature direction, 
        
        const double avg = igl::avg_edge_length(V,F);
        Eigen::Vector3d blue(0.2,0.2,0.8);
        viewer.data().add_edges(V + PD2*avg, V - PD2*avg, blue);
        
        // Set the viewer colors
        viewer.data().set_colors(colors_per_vertex);
    }

    if (key == '9') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        colors_per_vertex.setZero(V.rows(),3);
        // Add your code for computing the discrete maximal principal curvature:
        // store in K_max_principal
        Eigen::MatrixXd PD1,PD2;
        Eigen::VectorXd K_max_principal,PV2;
        igl::principal_curvature(V,F,PD1,PD2,K_max_principal,PV2);
        // For visualization, better to normalize the range of K_max_principal with the maximal and minimal curvatures
        // store colors in colors_per_vertex
        igl::jet(K_max_principal, K_max_principal.minCoeff(), K_max_principal.maxCoeff(), colors_per_vertex);
        // Uncomment the code below to draw a red segment parallel to the maximal curvature direction
        
        const double avg = igl::avg_edge_length(V,F);
        Eigen::Vector3d red(0.8,0.2,0.2);
        viewer.data().add_edges(V + PD1*avg, V - PD1*avg, red);
        
        // Set the viewer colors
        viewer.data().set_colors(colors_per_vertex);
    }

    if (key == 'E') {
        // Add your code for computing explicit Laplacian smoothing here:
        // store the smoothed vertices in V_expLap
        V_expLap = V;
        if(takeUniformL)
        {
            // Compute uniform Laplacian
            Eigen::SparseMatrix<double> A;
            igl::adjacency_matrix(F,A);
            Eigen::SparseVector<double> Asum;
            igl::sum(A,1,Asum);
            Eigen::SparseMatrix<double> Adiag, Adiaginv;
            igl::diag(Asum,Adiag);
            Eigen::SparseMatrix<double> Lu;
            Lu = A-Adiag;
            igl::invert_diag(Adiag, Adiaginv);
            for(int i = 0; i < Num_iter; ++i)
            {
                Eigen::MatrixXd translation;
                translation = (Adiaginv*Lu) * V_expLap;
                V_expLap += (Delta * translation);
            }
        }
        else
        {
            Eigen::SparseMatrix<double> Lc;
            igl::cotmatrix(V,F,Lc);
            Eigen::SparseMatrix<double> M,Minv;
            igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
            igl::invert_diag(M,Minv);
            // Compute cotangent Laplacian
            for(int i = 0; i < Num_iter; ++i)
            {
                Eigen::MatrixXd translation;
                translation = (Minv*Lc) * V_expLap;
                V_expLap += (Delta * translation);
            }
        }

        // Set the smoothed mesh
        viewer.data().clear();
        viewer.data().set_mesh(V_expLap, F);
    }

    if (key == 'D'){
        // Implicit smoothing for comparison
        // store the smoothed vertices in V_impLap
        V_impLap = V;
        if(!takeUniformL)
        {
            Eigen::SparseMatrix<double> L;
            igl::cotmatrix(V,F,L);
            for(int i = 0; i < Num_iter; ++i)
            {
                // Recompute just mass matrix on each step
                Eigen::SparseMatrix<double> M;
                igl::massmatrix(V_impLap,F,igl::MASSMATRIX_TYPE_BARYCENTRIC,M);
                // Solve (M-delta*L) U = M*U
                const auto & S = (M - Delta*L);
                Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
                assert(solver.info() == Eigen::Success);
                V_impLap = solver.solve(M*V_impLap).eval();
                // Compute centroid and subtract (also important for numerics)
                Eigen::VectorXd dblA;
                igl::doublearea(V_impLap,F,dblA);
                double area = 0.5*dblA.sum();
                Eigen::MatrixXd BC;
                igl::barycenter(V_impLap,F,BC);
                Eigen::RowVector3d centroid(0,0,0);
                for(int i = 0;i<BC.rows();i++)
                {
                  centroid += 0.5*dblA(i)/area*BC.row(i);
                }
                V_impLap.rowwise() -= centroid;
                // Normalize to unit surface area (important for numerics)
                V_impLap.array() /= sqrt(area);
            }
        }
        else
        {
            // Compute uniform Laplacian
            Eigen::SparseMatrix<double> A;
            igl::adjacency_matrix(F,A);
            Eigen::SparseVector<double> Asum;
            igl::sum(A,1,Asum);
            Eigen::SparseMatrix<double> Adiag, M;
            igl::diag(Asum,Adiag);
            Eigen::SparseMatrix<double> L;
            L = A-Adiag;
            igl::invert_diag(Adiag, M);
            for(int i = 0; i < Num_iter; ++i)
            {
                // Solve (M-delta*L) U = M*U
                const auto & S = (M - Delta*L);
                Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
                assert(solver.info() == Eigen::Success);
                V_impLap = solver.solve(M*V_impLap).eval();
                // Compute centroid and subtract (also important for numerics)
                Eigen::VectorXd dblA;
                igl::doublearea(V_impLap,F,dblA);
                double area = 0.5*dblA.sum();
                Eigen::MatrixXd BC;
                igl::barycenter(V_impLap,F,BC);
                Eigen::RowVector3d centroid(0,0,0);
                for(int i = 0;i<BC.rows();i++)
                {
                  centroid += 0.5*dblA(i)/area*BC.row(i);
                }
                V_impLap.rowwise() -= centroid;
                // Normalize to unit surface area (important for numerics)
                V_impLap.array() /= sqrt(area);
            }
        }
        // Set the smoothed mesh
        viewer.data().clear();
        viewer.data().set_mesh(V_impLap, F);
        //viewer.core().align_camera_center(V_impLap,F);
    }

    if (key == 'B') {
        // Add your code for computing bilateral smoothing here:
        // store the smoothed vertices in V_bilateral
        // be care of the sign mistake in the paper
        // use v' = v - n * (sum / normalizer) to update
        V_bilateral = V;
        std::vector<std::vector<int>> vertexAdjacency;
        igl::adjacency_list(F, vertexAdjacency);
        Eigen::MatrixXd normals, V_tmp;
        for(int it = 0; it < Num_iter; ++it)
        {
            V_tmp = V_bilateral;
            igl::per_vertex_normals(V_bilateral,F,normals);
            for(int v = 0; v < V.rows(); ++v)
            {
                Eigen::MatrixXd neighborMatrix;
                getKRingNeighborMatrixWithinSigmaC(vertexAdjacency,v,1,neighborMatrix);
                double sum = 0;
                double normalizer = 0;
                for(int i = 0; i < neighborMatrix.rows(); ++i)
                {
                    double t = (neighborMatrix.row(i) - V_bilateral.row(v)).norm();
                    double h = normals.row(v).dot((neighborMatrix.row(i) - V_bilateral.row(v)));
                    double wc = std::exp(-(t*t) / (2*sigmac*sigmac));
                    double ws = std::exp(-(h*h) / (2*sigmas*sigmas));
                    sum += wc*ws*h;
                    normalizer += wc*ws;
                }
                V_tmp.row(v) += (sum/normalizer) * normals.row(v);
            }
            V_bilateral = V_tmp;
        }
        // Set the smoothed mesh
        viewer.data().clear();
        viewer.data().set_mesh(V_bilateral, F);
    }
    return true;
}

bool load_mesh(Viewer& viewer,string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    if (filename.substr(filename.length() - 4) == ".off")
    {
        igl::readOFF(filename, V, F);
    }
    else if (filename.substr(filename.length() - 4) == ".obj")
    {
        igl::readOBJ(filename, V, F);
    }
    else
    {
        std::cerr << "Extension unknown (must be '.off' or '.obj')\n";
        return false;
    }
    viewer.data().clear();
    viewer.data().set_mesh(V,F);
    viewer.data().compute_normals();
    viewer.core().align_camera_center(V, F);
    return true;
}

int main(int argc, char *argv[]) {
    // Show the mesh
    Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    
    std::string filename;
    if (argc == 2) {
        filename = std::string(argv[1]);
    }
    else {
        filename = std::string("../data/bumpy-cube.obj");
    }
    load_mesh(viewer,filename,V,F);

    callback_key_down(viewer, '1', 0);

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // Expose variable directly ...
            ImGui::InputInt("k", &k, 0, 0);
            ImGui::InputInt("Num_iter", &Num_iter, 0, 0);
            ImGui::InputDouble("Delta", &Delta, 0, 0);
            ImGui::InputInt("takeUniformL", &takeUniformL, 0, 0);
            ImGui::InputDouble("SigmaC", &sigmac, 0, 0);
            ImGui::InputDouble("SigmaS", &sigmas, 0, 0);

        }
    };
    viewer.launch();
}
