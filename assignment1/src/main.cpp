#include <iostream>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/adjacency_list.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_corner_normals.h>
#include <igl/facet_components.h>
#include <igl/jet.h>
#include <igl/false_barycentric_subdivision.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/flip_edge.h>
#include <imgui/imgui.h>
/*** insert any libigl headers here ***/

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Vertex array, #V x3
Eigen::MatrixXd V;
// Face array, #F x3
Eigen::MatrixXi F;
// Per-face normal array, #F x3
Eigen::MatrixXd FN;
// Per-vertex normal array, #V x3
Eigen::MatrixXd VN;
// Per-corner normal array, (3#F) x3
Eigen::MatrixXd CN;
// Vectors of indices for adjacency relations
std::vector<std::vector<int> > VF, VFi, VV;
// Integer vector of component IDs per face, #F x1
Eigen::VectorXi cid;
// Per-face color array, #F x3
Eigen::MatrixXd component_colors_per_face;

void subdivide_sqrt3(const Eigen::MatrixXd &V,
					 const Eigen::MatrixXi &F,
					 Eigen::MatrixXd &Vout,
					 Eigen::MatrixXi &Fout)
{
    // Step 1: Put a vertex at the middle of each face and 
    // connect it to the three vertices of that face.
    igl::false_barycentric_subdivision(V, F, Vout, Fout);
    
    // Step 2: Move old vertices to new position
    igl::adjacency_list(F, VV, true);
	for (int i = 0; i < V.rows(); ++i) 
    {
        int n = VV[i].size();
        double a_n = (4 - 2 * cos(M_PI * 2 / n)) / 9;

        Eigen::Vector3d sum(0, 0, 0);
        for (auto neighbor : VV[i])
        {
            sum += V.row(neighbor);
        }

        Eigen::Vector3d v = V.row(i);
	    Eigen::Vector3d p = (1-a_n)*v + (a_n/n)*sum;
	    Vout.row(i) = p;
    }
    
    //Step 3: Flip the edges connecting points in P
    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP, B;
    std::vector<std::vector<int> > uE2E;
    igl::unique_edge_map(Fout, E, uE, EMAP, uE2E);
    for(int i = 0; i < uE.rows(); ++i)
    {
        if (uE(i, 0) < V.rows() && uE(i, 1) < V.rows())
        {
            // Don't flip edges that are on the boundary
            if (uE2E[i].size() == 2)
            {
                std::cout << i << std::endl;
                igl::flip_edge(Fout, E, uE, EMAP, uE2E, i);
            }
        }   
    }

}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to face relations here;
        // store in VF,VFi.
        igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);
        for(auto row : VF)
        {
            for(auto entry : row)
            {
                std::cout << entry << " ";
            }    
            std::cout << "\n";
        }
    }

    if (key == '2') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to vertex relations here:
        // store in VV.
        igl::adjacency_list(F, VV);
        for(auto row : VV)
        {
            for(auto entry : row)
            {
                std::cout << entry << " ";
            }    
            std::cout << "\n";
        }
    }

    if (key == '3') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        FN.setZero(F.rows(),3);
        // Add your code for computing per-face normals here: store in FN.
        igl::per_face_normals(V, F, FN);
        // Set the viewer normals.
        viewer.data().set_normals(FN);
    }

    if (key == '4') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing per-vertex normals here: store in VN.
        igl::per_vertex_normals(V, F, VN);
        // Set the viewer normals.
        viewer.data().set_normals(VN);
    }

    if (key == '5') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing per-corner normals here: store in CN.
        igl::per_corner_normals(V, F, 10, CN);
        //Set the viewer normals
        viewer.data().set_normals(CN);
    }

    if (key == '6') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        component_colors_per_face.setZero(F.rows(),3);
        // Add your code for computing per-face connected components here:
        // store the component labels in cid.
        std::vector<std::vector<std::vector<int>>> TT;
        Eigen::VectorXi components;
        igl::triangle_triangle_adjacency(F, TT);
        igl::facet_components(TT, cid, components);
        // Compute colors for the faces based on components, storing them in
        // component_colors_per_face.
        igl::jet(cid, true, component_colors_per_face);
        // Set the viewer colors
        viewer.data().set_colors(component_colors_per_face);

        for(int i = 0; i < components.rows(); ++i)
        {
            std::cout << "Component " << i << " has number of faces: " << components(i) << "\n";
        }
        std::cout << "The number of components is: " << components.size() << std::endl;
    }

    if (key == '7') {
		Eigen::MatrixXd Vout;
		Eigen::MatrixXi Fout;
        // Fill the subdivide_sqrt3() function with your code for sqrt(3) subdivision.
		subdivide_sqrt3(V,F,Vout,Fout);
        // Set up the viewer to display the new mesh
        V = Vout; F = Fout;
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
    }

    return true;
}

bool load_mesh(Viewer& viewer,string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
  igl::readOFF(filename,V,F);
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
        filename = std::string(argv[1]); // Mesh provided as command line argument
    }
    else {
        filename = std::string("../data/bunny.off"); // Default mesh
    }
	
    load_mesh(viewer,filename,V,F);

    callback_key_down(viewer, '1', 0);

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    
    viewer.launch();
}
