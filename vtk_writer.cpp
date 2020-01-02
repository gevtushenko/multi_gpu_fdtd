//
// Created by egi on 1/1/20.
//

#include "vtk_writer.h"

#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkSmartPointer.h>
#include <vtkRectilinearGrid.h>
#include <vtkXMLRectilinearGridWriter.h>

struct vtk_writer_impl
{
  vtkSmartPointer<vtkRectilinearGrid> grid;
  vtkSmartPointer<vtkFloatArray> x_array;
  vtkSmartPointer<vtkFloatArray> y_array;
  vtkSmartPointer<vtkFloatArray> z_array;
  vtkSmartPointer<vtkXMLRectilinearGridWriter> writer;
};

vtk_writer::vtk_writer (
  float dx_arg,
  float dy_arg,
  int nx_arg,
  int ny_arg,
  const std::string &filename_arg)
  : dx (dx_arg)
  , dy (dy_arg)
  , nx (nx_arg)
  , ny (ny_arg)
  , filename (filename_arg)
{

}

vtk_writer::~vtk_writer () = default;

void vtk_writer::write_vtu (
  const float *e)
{
  if (!p_impl)
    {
      p_impl = std::make_unique<vtk_writer_impl> ();

      p_impl->grid = vtkSmartPointer<vtkRectilinearGrid>::New ();
      p_impl->grid->SetDimensions (nx + 1, ny + 1, 1);

      p_impl->x_array = vtkSmartPointer<vtkFloatArray>::New ();

      for (int i = 0; i <= nx; i++)
        p_impl->x_array->InsertNextValue (dx * i);

      p_impl->y_array = vtkSmartPointer<vtkFloatArray>::New ();

      for (int i = 0; i <= ny; i++)
        p_impl->y_array->InsertNextValue (dy * i);

      p_impl->z_array = vtkSmartPointer<vtkFloatArray>::New ();
      p_impl->z_array->InsertNextValue (0.0f);

      p_impl->grid->SetXCoordinates (p_impl->x_array);
      p_impl->grid->SetYCoordinates (p_impl->y_array);
      p_impl->grid->SetZCoordinates (p_impl->z_array);

      p_impl->writer = vtkSmartPointer<vtkXMLRectilinearGridWriter>::New ();
    }

  vtkSmartPointer<vtkFloatArray> cell_data = vtkSmartPointer<vtkFloatArray>::New ();
  cell_data->SetName ("ez");

  for (int i = 0; i < nx * ny; i++)
    cell_data->InsertNextValue (e[i]);
  p_impl->grid->GetCellData()->SetScalars (cell_data);

  std::string step_file_name = filename + "_" + std::to_string (step++) + ".vtr";
  p_impl->writer->SetFileName (step_file_name.c_str ());
  p_impl->writer->SetInputData (p_impl->grid);
  p_impl->writer->Write ();
}

receiver_writer::receiver_writer (
  int time_steps_count_arg,
  int samples_count_arg)
  : time_steps_count (time_steps_count_arg)
  , samples_count (samples_count_arg)
  , values (new float[time_steps_count * samples_count])
  , writer (new vtk_writer (1.0, 0.1, samples_count, time_steps_count, "rx"))
{
  std::fill_n (values.get (), time_steps_count * samples_count, 0.0);
}

void receiver_writer::set_received_value (float value)
{
  values[(time_steps_count - 1 - time) * samples_count + sample] = value;
  writer->write_vtu (values.get ());

  time++;

  if (time == time_steps_count)
    {
      time = 0;
      sample++;
    }
}
