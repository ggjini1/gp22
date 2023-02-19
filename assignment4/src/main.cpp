#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>

/*** insert any necessary libigl headers here ***/
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/cat.h>
#include <igl/boundary_loop.h>
#include <igl/diag.h>
#include <igl/jet.h>
#include <igl/colormap.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

// vertex array, #V x3
Eigen::MatrixXd V;

// face array, #F x3
Eigen::MatrixXi F;

// UV coordinates, #V x2
Eigen::MatrixXd UV;

bool showingUV = false;
bool freeBoundary = false;
double TextureResolution = 10;
igl::opengl::ViewerCore temp3D;
igl::opengl::ViewerCore temp2D;
MatrixXd colors_per_face;
int distortion = 0;

void Redraw()
{
	viewer.data().clear();

	if (!showingUV)
	{
		viewer.data().set_mesh(V, F);
		viewer.data().set_face_based(false);
    	if(UV.size() != 0)
    	{
    	  viewer.data().set_uv(TextureResolution*UV);
    	  viewer.data().show_texture = true;
    	}
	}
	else
	{
		viewer.data().show_texture = false;
		viewer.data().set_mesh(UV, F);
	}
	if(colors_per_face.rows())
	{
    	viewer.data().set_colors(colors_per_face);
	}
}

bool callback_mouse_move(Viewer &viewer, int mouse_x, int mouse_y)
{
	if (showingUV)
		viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Translation;
	return false;
}

static void computeSurfaceGradientMatrix(SparseMatrix<double> & D1, SparseMatrix<double> & D2)
{
	MatrixXd F1, F2, F3;
	SparseMatrix<double> DD, Dx, Dy, Dz;

	igl::local_basis(V, F, F1, F2, F3);
	igl::grad(V, F, DD);

	Dx = DD.topLeftCorner(F.rows(), V.rows());
	Dy = DD.block(F.rows(), 0, F.rows(), V.rows());
	Dz = DD.bottomRightCorner(F.rows(), V.rows());

	D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
	D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
}
static inline void SSVD2x2(const Eigen::Matrix2d& J, Eigen::Matrix2d& U, Eigen::Matrix2d& S, Eigen::Matrix2d& V)
{
	double e = (J(0) + J(3))*0.5;
	double f = (J(0) - J(3))*0.5;
	double g = (J(1) + J(2))*0.5;
	double h = (J(1) - J(2))*0.5;
	double q = sqrt((e*e) + (h*h));
	double r = sqrt((f*f) + (g*g));
	double a1 = atan2(g, f);
	double a2 = atan2(h, e);
	double rho = (a2 - a1)*0.5;
	double phi = (a2 + a1)*0.5;

	S(0) = q + r;
	S(1) = 0;
	S(2) = 0;
	S(3) = q - r;

	double c = cos(phi);
	double s = sin(phi);
	U(0) = c;
	U(1) = s;
	U(2) = -s;
	U(3) = c;

	c = cos(rho);
	s = sin(rho);
	V(0) = c;
	V(1) = -s;
	V(2) = s;
	V(3) = c;
}

void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system
	// Hint: The matrix C should contain only one non-zero element per row and d should 
	// contain the positions in the correct order.
	// Build C
	int n = indices.rows();
	int m = V.rows();
	C.resize(2*n, 2*m);
	for(int i = 0; i < indices.rows(); ++i)
	{
		C.insert(i, indices(i)) = 1;
		C.insert(n+i, m+indices(i)) = 1;
	}
	// Build d
	d.resize(2*indices.rows(), 1);
	d = Map<VectorXd>(positions.data(), positions.cols()*positions.rows());
}

void computeParameterization(int type)
{
	VectorXi fixed_UV_indices;
	MatrixXd fixed_UV_positions;

	SparseMatrix<double> A;
	VectorXd b;
	Eigen::SparseMatrix<double> C;
	VectorXd d;
	// Find the indices of the boundary vertices of the mesh and put them in fixed_UV_indices
	if (!freeBoundary)
	{
		// The boundary vertices should be fixed to positions on the unit disc. Find these position and
		// save them in the #V x 2 matrix fixed_UV_position.
		igl::boundary_loop(F, fixed_UV_indices);
		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);
	}
	else
	{
		// Fix two UV vertices. This should be done in an intelligent way. 
		// Hint: The two fixed vertices should be the two most distant one on the mesh.
		VectorXd start = V.row(0);
		int maxIndex = 0;
		int maxDistanceSq = 0;
		for(int i = 0; i < V.rows(); ++i)
		{
			double distanceSq = (start.transpose() - V.row(i)).squaredNorm();
			if(distanceSq > maxDistanceSq)
			{
				maxDistanceSq = distanceSq;
				maxIndex = i;
			}	
		}
		fixed_UV_indices.resize(2);
		fixed_UV_indices << 0, maxIndex;
		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);
	}

	ConvertConstraintsToMatrixForm(fixed_UV_indices, fixed_UV_positions, C, d);

	// Find the linear system for the parameterization (1- Tutte, 2- Harmonic, 3- LSCM, 4- ARAP)
	// and put it in the matrix A.
	// The dimensions of A should be 2#V x 2#V.

	// The vector b is always zero
	b.resize(2*V.rows());
	b.setZero();

	if (type == '1') {
		// Add your code for computing uniform Laplacian for Tutte parameterization
		// Hint: use the adjacency matrix of the mesh
        // Compute uniform Laplacian
        SparseMatrix<double> Adjacency;
        igl::adjacency_matrix(F,Adjacency);
        SparseVector<double> AdjacencySum;
        igl::sum(Adjacency,1,AdjacencySum);
        SparseMatrix<double> AdjacencyDiag;
        igl::diag(AdjacencySum,AdjacencyDiag);
		SparseMatrix<double> Lu;
        Lu = Adjacency-AdjacencyDiag;
		igl::repdiag(Lu,2,A);
	}

	if (type == '2') {
		// Add your code for computing cotangent Laplacian for Harmonic parameterization
		// Use can use a function "cotmatrix" from libIGL, but ~~~~***READ THE DOCUMENTATION***~~~~
		SparseMatrix<double> L;
        igl::cotmatrix(V,F,L);
    	igl::repdiag(L,2,A);
	}

	if (type == '3') {
		// Add your code for computing the system for LSCM parameterization
		// Note that the libIGL implementation is different than what taught in the tutorial! Do not rely on it!!
		SparseMatrix<double> Du, Dv, Area;
		VectorXd AreaVector;
		// Compute area of triangles and store as diagonal matrix
		igl::doublearea(V,F,AreaVector);
		Area = AreaVector.asDiagonal();

		// Calculate the left hand side of our system to solve, i.e. XtX
		computeSurfaceGradientMatrix(Du, Dv);
		SparseMatrix<double> DutADu = 2 * Du.transpose() * Area * Du;
		SparseMatrix<double> DvtADv = 2 * Dv.transpose() * Area * Dv;
		SparseMatrix<double> DutADv = 2 * Du.transpose() * Area * Dv;
		SparseMatrix<double> DvtADu = 2 * Dv.transpose() * Area * Du;

		SparseMatrix<double> uleft = DutADu+DvtADv;
		SparseMatrix<double> uright = DvtADu-DutADv;
		SparseMatrix<double> lleft = DutADv-DvtADu;
		SparseMatrix<double> lright = DvtADv+DutADu;

		SparseMatrix<double> upper, lower;
		igl::cat(2, uleft, uright, upper);
		igl::cat(2, lleft, lright, lower);
		igl::cat(1, upper, lower, A);
	}

	if (type == '4') {
		// Add your code for computing ARAP system and right-hand side
		// Implement a function that computes the local step first
		// Then construct the matrix with the given rotation matrices
		Eigen::SparseMatrix<double> Dx, Dy;
		computeSurfaceGradientMatrix(Dx, Dy);
		VectorXd D1, D2, D3, D4;
		D1 = Dx * UV.col(0);
		D2 = Dy * UV.col(0);
		D3 = Dx * UV.col(1);
		D4 = Dy * UV.col(1);
		MatrixXd L; //< All rotation matrices combined
		L.resize(F.rows(), 4);
		for (int i = 0; i < F.rows(); i++) {
			Eigen::Matrix2d J, U, S, V, R, tmpD, VTr;
			J << D1[i], D2[i], D3[i], D4[i];
			SSVD2x2(J, U, S, V);
			R = U*V.transpose();
			L.row(i) << R(0, 0), R(0, 1), R(1, 0), R(1, 1);
		}
		// Compute area of triangles and store as diagonal matrix
		Eigen::SparseMatrix<double> Area;
		VectorXd AreaVector;
		igl::doublearea(V,F,AreaVector);
		Area = AreaVector.asDiagonal();
		// Set up the system matrices
		Eigen::SparseMatrix<double> uleft, uright, lleft, lright;
		uleft = (Dx.transpose()*Area*Dx + Dy.transpose()*Area*Dy);
		uright = MatrixXd::Zero(V.rows(), V.rows()).sparseView();
		lleft = MatrixXd::Zero(V.rows(), V.rows()).sparseView();
		lright = (Dx.transpose()*Area*Dx + Dy.transpose()*Area*Dy);
		SparseMatrix<double> upper, lower;
		igl::cat(2, uleft, uright, upper);
		igl::cat(2, lleft, lright, lower);
		igl::cat(1, upper, lower, A);
		VectorXd ub, lb;
		ub = Dx.transpose()*Area*L.col(0) + Dy.transpose()*Area*L.col(1);
		lb = Dx.transpose()*Area*L.col(2) + Dy.transpose()*Area*L.col(3);
		igl::cat(1, ub, lb, b);
	}

	// Solve the linear system.
	// Construct the system as discussed in class and the assignment sheet
	// Use igl::cat to concatenate matrices
	// Use Eigen::SparseLU to solve the system. Refer to tutorial 3 for more detail
	// Build left hand side matrix
	SparseMatrix<double> CT, upperLHS, lowerLHS, ZeroMat, LHS;
	ZeroMat.resize(C.rows(),C.rows());
	CT = C.transpose();
	igl::cat(2,A,CT,upperLHS);
	igl::cat(2,C,ZeroMat,lowerLHS);
	igl::cat(1,upperLHS,lowerLHS,LHS);
	// Build right hand side vector
	VectorXd RHS(b.rows()+d.rows());
	RHS << b,d;
	// Solve the system
	VectorXd x;
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	LHS.makeCompressed();
	solver.analyzePattern(LHS);
	solver.factorize(LHS);
	x = solver.solve(RHS);
	// The solver will output a vector
	UV.resize(V.rows(), 2);
	UV.col(0) = x.head(V.rows());
	UV.col(1) = x.head(2*V.rows()).tail(V.rows());
}

bool callback_key_pressed(Viewer &viewer, unsigned char key, int modifiers) {
	switch (key) {
	case '1':
	case '2':
	case '3':
	case '4':
		computeParameterization(key);
		break;
	case '5':
	{
		// Important: Assumes that Matrix UV is not empty
		VectorXd distortions(F.rows());
		colors_per_face.resize(F.rows(), 3);
		SparseMatrix<double> Dx, Dy;
		computeSurfaceGradientMatrix(Dx, Dy);
		VectorXd D1, D2, D3, D4;
		D1 = Dx * UV.col(0);
		D2 = Dy * UV.col(0);
		D3 = Dx * UV.col(1);
		D4 = Dy * UV.col(1);
		for(int i = 0; i < F.rows(); ++i)
		{
			// Compute the Jacobian of the current triangle J=left*right
			Eigen::Matrix2d J;
			J << D1[i], D2[i], D3[i], D4[i];
			// Compute distortion based on J
			if(distortion == 0)
			{
				// Conformal distortion measure
				JacobiSVD<Matrix2d> svd(J);
				auto SV = svd.singularValues();
				distortions(i) = pow(SV(0)-SV(1),2) / 2;
			}
			else if(distortion == 1)
			{
				// Isometric distortion measure
				JacobiSVD<Matrix2d> svd(J);
				auto SV = svd.singularValues();
				distortions(i) = pow(SV(0)-1,2) + pow(SV(1)-1,2);
			}
			else
			{
				// Area preserving distortion measure
				distortions(i) = pow(J.determinant()-1,2);
			}
		}

		// Create distortion colors from white to red
		for (int i = 0; i < colors_per_face.rows(); i++)
		{
			colors_per_face(i, 0) = 1;
			double interpol = 1 - (distortions(i) / distortions.maxCoeff());
			colors_per_face(i, 1) = interpol;
			colors_per_face(i, 2) = interpol;
			
		}
		break;
	}
	case '+':
		TextureResolution /= 2;
		break;
	case '-':
		TextureResolution *= 2;
		break;
	case ' ': // space bar -  switches view between mesh and parameterization
    if(showingUV)
    {
      temp2D = viewer.core();
      viewer.core() = temp3D;
      showingUV = false;
    }
    else
    {
      if(UV.rows() > 0)
      {
        temp3D = viewer.core();
        viewer.core() = temp2D;
        showingUV = true;
      }
      else { std::cout << "ERROR ! No valid parameterization\n"; }
    }
    break;
	}
	Redraw();
	return true;
}

bool load_mesh(string filename)
{
  igl::read_triangle_mesh(filename,V,F);
  Redraw();
  viewer.core().align_camera_center(V);
  showingUV = false;

  return true;
}

bool callback_init(Viewer &viewer)
{
	temp3D = viewer.core();
	temp2D = viewer.core();
	temp2D.orthographic = true;

	return false;
}

int main(int argc,char *argv[]) {
  if(argc != 2) {
    cout << "Usage ex4_bin <mesh.off/obj>" << endl;
    load_mesh("../data/cathead.obj");
  }
  else
  {
    // Read points and normals
    load_mesh(argv[1]);
  }

	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("Parmaterization", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::Checkbox("Free boundary", &freeBoundary);
			ImGui::InputInt("Distortion (0,1,2)", &distortion);
			// TODO: Add more parameters to tweak here...
		}
	};

  viewer.callback_key_pressed = callback_key_pressed;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_init = callback_init;

  viewer.launch();
}
