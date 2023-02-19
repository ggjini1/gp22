#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/jet.h>
#include <igl/slice.h>
#include <igl/writeOFF.h>

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Normals evaluated via PCA method, #P x3
Eigen::MatrixXd NP;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
int polyDegree = 2;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 1.0;

// Parameter: grid resolution
int resolution = 20;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// The spatial index datastructure
std::map<unsigned int, std::vector<int>> gridSI;
// The size of the spatial index datastructure 
int nx, ny, nz;
double xmax, xmin, ymax, ymin, zmax, zmin;

// Functions
void createGrid();
void evaluateImplicitFunc();
void evaluateImplicitFunc_PolygonSoup();
void getLines();
void pcaNormal();
bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers);

// Given a 3d point, returns the spatial index of the grid in which it is located
// Precondition: Global variables nx,ny,nz and {x,y,z}_min and {x,y,z}_max need to be computed
unsigned int getSpatialIndex(const Eigen::Vector3d& point)
{
    double x = point(0);
    double y = point(1);
    double z = point(2);
    int xpos = static_cast<int>(std::floor((x-xmin)/wendlandRadius));
    int ypos = static_cast<int>(std::floor((y-ymin)/wendlandRadius));
    int zpos = static_cast<int>(std::floor((z-zmin)/wendlandRadius));

    int index = nx*ny*zpos + nx*ypos + xpos;
    return index;
}

// Precpomputes everything that is necessary for spatial index queries
void setupSpatialIndexStructure(const Eigen::MatrixXd& Points)
{
    // Max and min in every direction
    Eigen::VectorXd minCoeffs = Points.colwise().minCoeff();
    Eigen::VectorXd maxCoeffs = Points.colwise().maxCoeff();
    xmin = minCoeffs(0) - 10;
    ymin = minCoeffs(1) - 10;
    zmin = minCoeffs(2) - 10;
    xmax = maxCoeffs(0) + 10;
    ymax = maxCoeffs(1) + 10;
    zmax = maxCoeffs(2) + 10;

    // Compute number of cells in each direction
    nx = static_cast<int>(std::ceil((xmax-xmin)/wendlandRadius));
    ny = static_cast<int>(std::ceil((ymax-ymin)/wendlandRadius));
    nz = static_cast<int>(std::ceil((zmax-zmin)/wendlandRadius));

    // Fill the datastructure with our points
    std::map<unsigned int, std::vector<int>> localgridSI;
    for (unsigned int i = 0; i < Points.rows(); ++i)
    {
        unsigned int index = getSpatialIndex(Points.row(i));
        localgridSI[index].push_back(i);
    }
    gridSI = localgridSI;
}

int closestPointBruteForce(const Eigen::Vector3d& point)
{
    double closestDistance=10000000000000;
    int closestIndex;
    for (int i = 0; i < P.rows(); ++i)
    {
        double distance = (point.transpose() - P.row(i)).squaredNorm();
        if (distance < closestDistance)
        {
            closestDistance = distance;
            closestIndex = i;
        }
    }
    return closestIndex;
}

int closestPoint(const Eigen::Vector3d& point)
{
    unsigned int index = getSpatialIndex(point);
    double closestDistance=10000000000000;
    int closestIndex;
    if (gridSI.find(index) == gridSI.end())
    {
        return closestPointBruteForce(point);
    }
    for (int i = 0; i < gridSI[index].size(); ++i)
    {
        int otherpointIndex = gridSI[index][i];
        double distance = (point.transpose() - P.row(otherpointIndex)).squaredNorm();
        if (distance < closestDistance)
        {
            closestDistance = distance;
            closestIndex = otherpointIndex;
        }
    }
    return closestIndex;
}

// Brute Force
// Only call after having computed the constrained points
void getClosestPointsWithinHBruteForce(const Eigen::Vector3d& point, const Eigen::MatrixXd& pointcloud, std::vector<int>& neighborIndex)
{
    for (unsigned int i = 0; i < pointcloud.rows(); ++i)
    {
        double distance = (point.transpose() - pointcloud.row(i)).squaredNorm();
        if (distance <= wendlandRadius*wendlandRadius)
        {
            neighborIndex.push_back(i);
        }
    }
}

// Only call after having computed the constrained points
void getClosestPointsWithinH(const Eigen::Vector3d& point, const Eigen::MatrixXd& pointcloud, std::vector<int>& neighborIndex)
{
    int xpos = static_cast<int>(std::floor((point(0)-xmin)/wendlandRadius));
    int ypos = static_cast<int>(std::floor((point(1)-ymin)/wendlandRadius));
    int zpos = static_cast<int>(std::floor((point(2)-zmin)/wendlandRadius));

    for (int i = -1; i < 2; ++i)
    {
        if (0 > xpos+i || xpos+i >= nx)
            continue;
        for (int j = -1; j < 2; ++j)
        {
            if (0 > ypos+j || ypos+j >= ny)
                continue;
            for (int k = -1; k < 2; ++k)
            {
                if (0 > zpos+k || zpos+k >= nz)
                    continue;
                int index = nx*ny*(zpos+k) + nx*(ypos+j) + (xpos+i);
                for (auto& neighbors : gridSI[index])
                {
                    double distance = (point.transpose() - pointcloud.row(neighbors)).squaredNorm();
                    if (distance <= wendlandRadius*wendlandRadius)
                    {
                        neighborIndex.push_back(neighbors);
                    }
                }
            }
        }
    }
}

void setUpPolynomialMatrix(const Eigen::MatrixXd& Points, Eigen::MatrixXd& polynomial)
{
    for(unsigned int i = 0; i < Points.rows(); ++i)
    {
        double px, py, pz;
        px = Points(i,0); py = Points(i,1); pz = Points(i,2);
        switch (polyDegree)
        {
        case 0:
        {
            polynomial.row(i) << 1;
            break;
        }
        case 1:
        {
            polynomial.row(i) << 1, px, py, pz;
            break;
        }
        default:
            polynomial.row(i) << 1, px, py, pz, pow(px, 2), pow(py, 2), pow(pz, 2), px* py, py* pz, pz* px;
            break;
        }
    }
}

// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid()
{
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines.resize(0, 6);
    grid_values.resize(0);
    V.resize(0, 3);
    F.resize(0, 3);
    FN.resize(0, 3);

    Eigen::MatrixXd centered = P.rowwise() - P.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(P.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(cov);
	Eigen::MatrixXd eigenVectors = eigen.eigenvectors();
    if (eigenVectors.determinant() < 0)
    {
		eigenVectors = eigenVectors * (-1);
    } 
	Eigen::MatrixXd transform = eigenVectors.rightCols(3);
    Eigen::MatrixXd rotatedP = centered * transform;

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = rotatedP.colwise().minCoeff();
    bb_max = rotatedP.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x)
    {
        for (unsigned int y = 0; y < resolution; ++y)
        {
            for (unsigned int z = 0; z < resolution; ++z)
            {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
    grid_points = grid_points * transform.adjoint();
	grid_points = grid_points.rowwise() + P.colwise().mean();
}

// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS
void evaluateImplicitFunc()
{
    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    // Evaluate sphere's signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x)
    {
        for (unsigned int y = 0; y < resolution; ++y)
        {
            for (unsigned int z = 0; z < resolution; ++z)
            {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                Eigen::Vector3d gridPoint = grid_points.row(index);
                double px, py, pz;
                px = gridPoint(0); py = gridPoint(1); pz = gridPoint(2);
                
                // Determine closest point to grid point
                std::vector<int> neighborIndex;
                std::vector<int> neighborIndexBF;
                getClosestPointsWithinH(gridPoint, constrained_points, neighborIndex);
                if (neighborIndex.size() == 0)
                {
                    grid_values[index] = 1000;
                    continue;
                }

                // Slice the matrices to get neighbor points
                Eigen::MatrixXd neighborPoints;
                Eigen::VectorXd neighborValues;
                Eigen::VectorXi neighborIndexVec = Eigen::VectorXi::Map(neighborIndex.data(), neighborIndex.size());
                igl::slice(constrained_points, neighborIndexVec, 1, neighborPoints);
                igl::slice(constrained_values, neighborIndexVec, neighborValues);

                // Set up weight matrix
                Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(neighborPoints.rows(), neighborPoints.rows());
                for (unsigned int i = 0; i < neighborPoints.rows(); i++)
                {
                    // Distance between grid point and the neighbor
                    double distance = (gridPoint.transpose() - neighborPoints.row(i)).norm();

                    // Compute weight for current neighbor
                    double weightNeighbor = pow((1 - (distance / wendlandRadius)), 4) * ((4 * distance / wendlandRadius) + 1);
                    weights(i, i) = weightNeighbor;
                }

                // Set up polyonmial Matrix
                Eigen::MatrixXd polynomial;
                Eigen::VectorXd bx;
                switch (polyDegree)
                {
                case 0:
                {
                    polynomial.resize(neighborPoints.rows(), 1);
                    bx.resize(1); bx << 1;
                    break;
                }
                case 1:
                {
                    polynomial.resize(neighborPoints.rows(), 4);
                    bx.resize(4); bx << 1, px, py, pz;
                    break;
                }
                default:
                    polynomial.resize(neighborPoints.rows(), 10);
                    bx.resize(10); bx << 1, px, py, pz, pow(px, 2), pow(py, 2), pow(pz, 2), px* py, py* pz, pz* px;
                    break;
                }
                setUpPolynomialMatrix(neighborPoints, polynomial);

                // Solve the system
                Eigen::MatrixXd A = weights * polynomial;
                Eigen::VectorXd B = weights * neighborValues;
                Eigen::VectorXd c = A.colPivHouseholderQr().solve(B);
                // Value at (x,y,z) = implicit function for the sphere
                grid_values[index] = c.dot(bx);
            }
        }
    }
}

void evaluateImplicitFunc_PolygonSoup()
{
    // Replace with your code here, for "key == '5'"
    setupSpatialIndexStructure(P);

    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    // Evaluate sphere's signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x)
    {
        for (unsigned int y = 0; y < resolution; ++y)
        {
            for (unsigned int z = 0; z < resolution; ++z)
            {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                Eigen::Vector3d gridPoint = grid_points.row(index);
                double px, py, pz;
                px = gridPoint(0); py = gridPoint(1); pz = gridPoint(2);
                
                // Determine closest point to grid point
                std::vector<int> neighborIndex;
                getClosestPointsWithinH(gridPoint, P, neighborIndex);
                if (neighborIndex.size() == 0)
                {
                    grid_values[index] = 1000;
                    continue;
                }

                // Slice the matrices to get neighbor points
                Eigen::MatrixXd neighborPoints;
                Eigen::VectorXi neighborIndexVec = Eigen::VectorXi::Map(neighborIndex.data(), neighborIndex.size());
                igl::slice(P, neighborIndexVec, 1, neighborPoints);

                // Compute the values from the constraints
                Eigen::VectorXd neighborValues(neighborIndex.size());
                for (int i = 0; i < neighborIndex.size(); ++i)
                {
                    Eigen::Vector3d neighboringPoint = neighborPoints.row(i);
                    neighborValues(i) = ((gridPoint - neighboringPoint).transpose()).dot(N.row(neighborIndex[i]));
                }

                // Set up weight matrix
                Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(neighborPoints.rows(), neighborPoints.rows());
                for (unsigned int i = 0; i < neighborPoints.rows(); i++)
                {
                    // Distance between grid point and the neighbor
                    double distance = (gridPoint.transpose() - neighborPoints.row(i)).norm();

                    // Compute weight for current neighbor
                    double weightNeighbor = pow((1 - (distance / wendlandRadius)), 4) * ((4 * distance / wendlandRadius) + 1);
                    weights(i, i) = weightNeighbor;
                }

                // Set up polyonmial Matrix
                Eigen::MatrixXd polynomial;
                Eigen::VectorXd bx;
                switch (polyDegree)
                {
                case 0:
                {
                    polynomial.resize(neighborPoints.rows(), 1);
                    bx.resize(1); bx << 1;
                    break;
                }
                case 1:
                {
                    polynomial.resize(neighborPoints.rows(), 4);
                    bx.resize(4); bx << 1, px, py, pz;
                    break;
                }
                default:
                    polynomial.resize(neighborPoints.rows(), 10);
                    bx.resize(10); bx << 1, px, py, pz, pow(px, 2), pow(py, 2), pow(pz, 2), px* py, py* pz, pz* px;
                    break;
                }
                setUpPolynomialMatrix(neighborPoints, polynomial);

                // Solve the system
                Eigen::MatrixXd A = weights * polynomial;
                Eigen::VectorXd B = weights * neighborValues;
                Eigen::VectorXd c = A.colPivHouseholderQr().solve(B);
                // Value at (x,y,z) = implicit function for the sphere
                grid_values[index] = c.dot(bx);
            }
        }
    }
}

// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines()
{
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x < resolution; ++x)
    {
        for (unsigned int y = 0; y < resolution; ++y)
        {
            for (unsigned int z = 0; z < resolution; ++z)
            {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1)
                {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1)
                {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1)
                {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

// Estimation of the normals via PCA.
void pcaNormal()
{
    // Perform PCA
    Eigen::MatrixXd centered = P.rowwise() - P.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(P.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(cov);
	Eigen::MatrixXd eigenVectors = eigen.eigenvectors();
    Eigen::VectorXd eigenValues = eigen.eigenvalues();

    // Find index of largest eigenvalue
    double maxEV = 0;
    int indexMax = -1;
    for (int i = 0; i < eigenValues.size(); ++i)
    {
        if (eigenValues(i) > maxEV)
        {
            maxEV = eigenValues(i);
            indexMax = i;
        }
    }

    // Compute perpendicular vector to largest eigenvector, which is norm
    Eigen::Vector3d norm;
    norm << -eigenVectors(1, indexMax), eigenVectors(0, indexMax), eigenVectors(2, indexMax);

    // Set the estimated norms
    NP.resize(N.rows(), N.cols());
    for (int i = 0; i < N.rows(); ++i)
    {
        double scalarProduct = N.row(i).dot(norm);
        if (scalarProduct > 0)
        {
            NP.row(i) = norm;
        }
        else
        {
            NP.row(i) = norm * (-1);
        }
    }
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers)
{
    if (key == '1')
    {
        // Show imported points
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        viewer.data().point_size = 11;
        viewer.data().add_points(P, Eigen::RowVector3d(0, 0, 0));
    }

    if (key == '2')
    {
        // Create spatial index data structure
        setupSpatialIndexStructure(P);
        // Show all constraints
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        // Add your code for computing auxiliary constraint points here
        double eps = 0.01 * igl::bounding_box_diagonal(P);
        constrained_points.resize(3*P.rows(), 3);
        constrained_values.resize(3*P.rows(), 1);
        for (unsigned int i = 0; i < P.rows(); ++i)
        {
            double local_eps_plus = eps;
            Eigen::Vector3d ppluseps = P.row(i) + eps*N.row(i);
            while (closestPoint(ppluseps) != i)
            {
                local_eps_plus = local_eps_plus / 2;
                ppluseps = P.row(i) + local_eps_plus*N.row(i);
            }
            double local_eps_minus = eps;
            Eigen::Vector3d pminuseps = P.row(i) - eps*N.row(i);
            while (closestPoint(pminuseps) != i)
            {
                local_eps_minus = local_eps_minus / 2;
                pminuseps = P.row(i) - local_eps_minus*N.row(i);
            }
            constrained_points.row(i*3 + 0) = ppluseps;
            constrained_points.row(i*3 + 1) = pminuseps;
            constrained_points.row(i*3 + 2) = P.row(i);
            constrained_values(i*3 + 0) = local_eps_plus;
            constrained_values(i*3 + 1) = -local_eps_minus;
            constrained_values(i*3 + 2) = 0;
        }
        // Add code for displaying all points, as above
        Eigen::MatrixXd colorMap;
        colorMap.setZero(constrained_points.rows(), 3);
        for (int i = 0; i < constrained_points.rows(); ++i)
        {
            double value = constrained_values(i);
            if (value < 0)
            {
                colorMap(i, 1) = 1;
            }
            else
            {
                if (value > 0)
                    colorMap(i, 0) = 1;
            }
        }
        viewer.data().add_points(constrained_points, colorMap);
    }

    if (key == '3')
    {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        // Add code for creating a grid
        // Add your code for evaluating the implicit function at the grid points
        // Add code for displaying points and lines
        // You can use the following example:

        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Make grid
        createGrid();

        // Create spatial index data structure
        setupSpatialIndexStructure(constrained_points);

        // Evaluate implicit function
        evaluateImplicitFunc();

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i)
        {
            double value = grid_values(i);
            if (value < 0)
            {
                grid_colors(i, 1) = 1;
            }
            else
            {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                                grid_lines.block(0, 3, grid_lines.rows(), 3),
                                Eigen::RowVector3d(0.8, 0.8, 0.8));
        /*** end: sphere example ***/
    }

    if (key == '4')
    {
        // Show reconstructed mesh
        viewer.data().clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0))
        {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0)
        {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        igl::per_face_normals(V, F, FN);
        viewer.data().set_mesh(V, F);
        viewer.data().show_lines = true;
        viewer.data().show_faces = true;
        viewer.data().set_normals(FN);
    }

    if (key == '5')
    {
        // Use the structure for key=='3' but replace the function evaluateImplicitFunc();
        // with a function performing the approximation of the implicit surface from polygon soup
        // Ref: Chen Shen, James F. O’Brien, and Jonathan Richard Shewchuk. Interpolating and approximating implicit surfaces from polygon soup.

        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core().align_camera_center(P);

        // Make grid
        createGrid();

        // Evaluate implicit function --> Function to be modified here
        evaluateImplicitFunc_PolygonSoup();

        // get grid lines
        getLines();

        // Display the reconstruction
        callback_key_down(viewer, '4', modifiers);
    }

    if (key == '6' || key == '7' || key == '8')
    {
        // Implement PCA Normal Estimation --> Function to be modified here
        pcaNormal();

        // To use the normals estimated via PCA instead of the input normals and then restaurate the input normals
        Eigen::MatrixXd N_tmp = N;
        N = NP;

        switch (key)
        {
        case '6':
            callback_key_down(viewer, '2', modifiers);
            break;
        case '7':
            callback_key_down(viewer, '3', modifiers);
            break;
        case '8':
            callback_key_down(viewer, '3', modifiers);
            callback_key_down(viewer, '4', modifiers);
            break;
        default:
            break;
        }

        // Restore input normals
        N = N_tmp;
    }

    return true;
}

bool callback_load_mesh(Viewer &viewer, string filename)
{
    igl::readOFF(filename, P, F, N);
    callback_key_down(viewer, '1', 0);
    return true;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage ex2_bin <mesh.off>" << endl;
        igl::readOFF("../data/sphere.off", P, F, N);
    }
    else
    {
        // Read points and normals
        igl::readOFF(argv[1], P, F, N);
    }

    Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    viewer.callback_key_down = callback_key_down;

    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // Expose variable directly ...
            ImGui::InputInt("Resolution", &resolution, 0, 0);
            ImGui::InputInt("Polynomial degree", &polyDegree, 0, 0);
            ImGui::InputDouble("Wendland Radius", &wendlandRadius, 0, 0);
            if (ImGui::Button("Reset Grid", ImVec2(-1, 0)))
            {
                std::cout << "ResetGrid\n";
                // Recreate the grid
                createGrid();
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }

            // TODO: Add more parameters to tweak here...
        }
    };

    callback_key_down(viewer, '1', 0);

    viewer.launch();
}
