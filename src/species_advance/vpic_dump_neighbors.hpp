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

// Create a functor to log the neighbor indices.
// The output file will simply be a column of all
// the neighbor indices encountered in move_p_kokkos.
template<typename index_t>
struct DUMP_NEIGHBORS
{
  // Constructor to tell which file to dump to and construct
  // an output stream for the file. By default, everything is
  // appended!
  DUMP_NEIGHBORS(const std::string & _ifile,
                 const std::string & _pfile) : index_file(_ifile), 
                                               planes_file(_pfile),
                                               ifile_stream(index_file, std::ofstream::app),
                                               pfile_stream(planes_file, std::ofstream::app)
  {}

  void operator () (const index_t value)
  {
    ifile_stream << value << "\n";
  }

  void write_planes( const index_t xval, const index_t yval, const index_t zval )
  {
    pfile_stream << xval << ", " << yval << ", " << zval << "\n";
  }

  ~DUMP_NEIGHBORS()
  { 
    ifile_stream.close(); 
    pfile_stream.close();
  }

  private:
    const std::string index_file;
    const std::string planes_file;
    std::ofstream ifile_stream;
    std::ofstream pfile_stream;
};



