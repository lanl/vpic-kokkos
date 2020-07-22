/* *****************************************************************
 *  This header is used to dump the neighbor indices easily
 *  for testing. To use it, VPIC must be built with 
 *  VPIC_DUMP_NEIGHBORS=ON. The neighbor indices will then be 
 *  appended to the file "index_file".
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
  DUMP_NEIGHBORS(const std::string & file) : index_file(file), 
                                             file_stream(index_file, std::ofstream::app)
  {}

  void operator () (const index_t value)
  {
    file_stream << value << "\n";
  }

  ~DUMP_NEIGHBORS(){ file_stream.close(); }

  private:
    const std::string index_file;
    std::ofstream file_stream;
};



