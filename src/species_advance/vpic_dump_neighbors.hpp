/* *****************************************************************
 *  This header is used to dump the neighbor indices easily
 *  for testing. To use it, VPIC must be built with 
 *  VPIC_DUMP_NEIGHBORS=ON. The neighbor indices will then be 
 *  appended to the file "index_file". The planes {-1, 0, 1} 
 *  leading to that index is in "planes_file".
 *
 *  Author: W. Joe Meese
 * **************************************************************** */

#include <fstream>
#include <string>
#include <cmath> // needed for abs()

static const std::string neighbor_types [4] = { "Self", "Face", "Edge", "Corner" };

// Create a functor to log the neighbor indices.
// The output file will simply be a column of all
// the neighbor indices encountered in move_p_kokkos.
template<typename index_t, typename pos_t>
struct DUMP_NEIGHBORS
{
  // Constructor to tell which file to dump to and construct
  // an output stream for the file. By default, everything is
  // appended!
  DUMP_NEIGHBORS(const std::string & _ifile,
                 const std::string & _pfile,
                 const std::string & _posfile) : index_file(_ifile), 
                                                 planes_file(_pfile),
                                                 pos_file(_posfile),
                                                 ifile_stream(index_file, std::ofstream::app),
                                                 pfile_stream(planes_file, std::ofstream::app),
                                                 posfile_stream(pos_file, std::ofstream::app)
  {}

  void operator () (const index_t value)
  {
    ifile_stream << value << "\n";
  }

  void write_planes( const index_t xval, const index_t yval, const index_t zval )
  {
    pfile_stream << xval << ", " << yval << ", " << zval << "\n";
  }

  void write_final_cell( const pos_t xpos, const pos_t ypos, const pos_t zpos )
  {
      // Exit if the final cell has already been determined
      // for this particle in move_p_kokkos
      if (checked) return;

      // If count = 1, the final cell is connected at a face.
      // If count = 2, the final cell is connected at an edge.
      // If count = 3, the final cell is connected at a corner.
      int count = 0;
      if ( abs(xpos) > (pos_t) 1 ) ++count;
      if ( abs(ypos) > (pos_t) 1 ) ++count;
      if ( abs(zpos) > (pos_t) 1 ) ++count;

      posfile_stream << neighbor_types[count] << "\n";
      
      checked = true;
  }

  ~DUMP_NEIGHBORS()
  { 
    ifile_stream.close(); 
    pfile_stream.close();
    posfile_stream.close();
  }

  private:
    bool checked = false;             // This boolean eliminates double-counting
    const std::string index_file;     // File name for the index_file
    const std::string planes_file;    // File name for the planes_file
    const std::string pos_file;       // File name for the planes_file
    std::ofstream ifile_stream;       // File stream for the indices
    std::ofstream pfile_stream;       // File stream for the planes
    std::ofstream posfile_stream;     // File stream for the final positions

};



