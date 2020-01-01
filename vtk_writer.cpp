//
// Created by egi on 1/1/20.
//

#include "vtk_writer.h"

#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkSmartPointer.h>
#include <vtkRectilinearGrid.h>
#include <vtkXMLRectilinearGridWriter.h>

void write_vtu (
  const std::string &filename,
  float dx,
  float dy,
  int nx,
  int ny,
  const float *e)
{
  vtkSmartPointer<vtkRectilinearGrid> grid = vtkSmartPointer<vtkRectilinearGrid>::New ();
  grid->SetDimensions (nx + 1, ny + 1, 1);

  vtkSmartPointer<vtkFloatArray> x_array = vtkSmartPointer<vtkFloatArray>::New ();

  for (int i = 0; i <= nx; i++)
    x_array->InsertNextValue (dx * i);

  vtkSmartPointer<vtkFloatArray> y_array = vtkSmartPointer<vtkFloatArray>::New ();

  for (int i = 0; i <= ny; i++)
    y_array->InsertNextValue (dy * i);

  vtkSmartPointer<vtkFloatArray> z_array = vtkSmartPointer<vtkFloatArray>::New ();
  z_array->InsertNextValue (0.0f);

  grid->SetXCoordinates (x_array);
  grid->SetYCoordinates (y_array);
  grid->SetZCoordinates (z_array);

  vtkSmartPointer<vtkFloatArray> cell_data = vtkSmartPointer<vtkFloatArray>::New ();
  cell_data->SetName ("ez");

  for (int i = 0; i < nx * ny; i++)
    cell_data->InsertNextValue (e[i]);
  grid->GetCellData()->SetScalars (cell_data);

  vtkSmartPointer<vtkXMLRectilinearGridWriter> writer = vtkSmartPointer<vtkXMLRectilinearGridWriter>::New ();
  writer->SetFileName (filename.c_str ());
  writer->SetInputData (grid);
  writer->Write ();
}
