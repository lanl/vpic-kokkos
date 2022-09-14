
// Utility to parse lost particle data into angle-resolved spectra
//
// Usage: 
//     ./anglehist [species name] optional: [maximum energy in MeV]
//
//
// icc -fopenmp -o anglehist anglehist.c
//
// Requires OpenMP 4.5 (gcc 6.1 or Intel 17.0), or you may run slowrially
//
// This program must match how the lost particle data are written as defined in
// your deck.  TODO: Update this all to hdf5 or similar, so you don't have to
// know the data layout bit for bit.
//
// 2Dbinhistplot.py will make a nice plot for you using the output of this
// program.
//
// All the particles with energies above the maximum of the histogram will be
// artificially placed in the highest bin.
//
// First release written by Scott V. Luedtke, XCP-6, October 4, 2019.

#include <stdio.h>
#include <stdlib.h>   /* for atoi */
#include <stdarg.h>   /* for va_list, va_start, va_end */
#include <errno.h>
#include <string.h>   /* for strcspn */ 
#include <math.h>     /* for sqrt */
#include <sys/stat.h> /* for mkdir */
#include <stdint.h>   /* for uint32_t, uint64_t */
#include <inttypes.h> /* to print uint64_t */
#include <glob.h>     /* to get list of filenames */

#define BEGIN_PRIMITIVE do
#define END_PRIMITIVE   while (0)

void print_log( const char *fmt, ... );
#define ERROR(args) BEGIN_PRIMITIVE {                                         \
 print_log( "Error at %s(%i):\n\t", __FILE__, __LINE__ );                     \
 print_log args;                                                              \
 print_log( "\n" );                                                           \
} END_PRIMITIVE
#define ABORT(cond) { if (cond) exit(0); }

//---------------------------------------------------------------------
// General purpose memory allocation macro 
#define ALLOCATE(A,LEN,TYPE)                                                  \
  if ( !((A)=(TYPE *)malloc((size_t)(LEN)*sizeof(TYPE))) )                    \
      ERROR(("Cannot allocate."));

// Construct an index for the particle data.  This must match how the data are
// output in the lost particle processor in your VPIC deck.
#define ux 0
#define uy 1
#define uz 2
#define w 6
#define x 3
#define y 4
#define z 5
#define numvars 7

int main( int argc, char *argv[] ) { 
  fprintf(stderr,"Ham.\n");
  int num_tracers_total, nprocs; 
  int nvar; 
  int itmax; 
  char temp[256];
  char usage_msg[] = 
             "Usage: ./lostspec [species name] optional: [maximum energy in MeV]\n\n";

  if ( argc != 2 && argc != 3) { 
    fprintf( stderr, "%s", usage_msg ); 
    exit(0); 
  } 

  // Read some numbers from params.txt
  char buffer[1024];
  //int interval, nstep_total, i;
  FILE *params;
  params = fopen("../params.txt", "r");
  if (!params) ERROR(("Cannot open params.txt.  (The location is probably wrong.)"));

  double timeToSI, lengthToSI, massToSI, chargeToSI;
  fgets(buffer, 1024, params);
  fscanf(params, "%lf %[^\n]\n", &timeToSI, buffer);
  fscanf(params, "%lf %[^\n]\n", &lengthToSI, buffer);
  fscanf(params, "%lf %[^\n]\n", &massToSI, buffer);
  fscanf(params, "%lf %[^\n]\n", &chargeToSI, buffer);
  fclose(params);

  double emin = 0;
  double emax; // MeV
  double Ecut = 0;

  char *particle = argv[1];
  if (argc>2) emax = atof(argv[2]);
  else emax = 500;

  double amax = M_PI;
  double amin = 0;

  // Must define these to use in the reduction clause
#define nbinse 100
#define nbinsa 100
  double de = emax/(double)nbinse;
  double da = amax/(double)nbinsa;
  // When you have 10^13 particles at 10^8 precision, you might need extended
  // precision for the sums.
  long double hist[nbinsa][nbinse] = {0};

  // The code uses normalized momentum, so don't use the conversion factors
  // from params.txt.  If you want to use different units than here, change
  // this, or manually adjust them in your plotter.
#define e_SI (1.602176634e-19) /* C */
#define c_SI (2.99792458e8) /* m / s */
#define m_e_SI (9.1093837015e-31) /* kg */
#define mp_me 1836.15267343
  double ekMeVconst;
  double elecekMeVconst = m_e_SI*c_SI*c_SI*1e-6/e_SI;
  double carbekMeVconst = 12.*mp_me*elecekMeVconst;
  double protonekMeVconst = mp_me*elecekMeVconst;
  if (strcmp(particle, "I2")==0) ekMeVconst = carbekMeVconst;
  else if (strcmp(particle, "proton")==0){
      ekMeVconst = protonekMeVconst;
      particle = "I2";
  }
  else ekMeVconst = elecekMeVconst;

  //TODO: Race this (untested!) bit of code against glob on a large VPIC run on
  //Lustre

  //// Count how many files there are in the lostparts directory
  //char dirname[256] = "../../lostparts";
  //struct dirent *de;
  //DIR *dir = opendir(dirname);
  //if(!dir) ERROR("Directory %s not found", dirname);
  //int count=0;
  //while (de = readdir(dir)) ++count;
  //closedir(dir);

  //// Construct the (massive) list of files
  //char **filelist;
  //ALLOCATE(filelist, count, char*);
  //dir = opendir(dirname);
  //for(int i=0;i<count;i++){
  //   ALLOCATE(filelist[i], 64, char);
  //   filelist[i] = readdir(dir)->d_name;
  //}


  char filepath[256]; // Better be big enough
  sprintf(filepath, "../pb_diagnostic/%s.*", particle);

  glob_t globbuf;
  glob(filepath, GLOB_NOSORT, NULL, &globbuf); 
  size_t count = globbuf.gl_pathc;
  fprintf(stderr, "Found %zu\n files", count);

  // Consider only particles within a certain box
  // Check if the line that does this is commented below!
  double xmin = -30e-6/lengthToSI;
  double xmax = 70e-6/lengthToSI;
  double ymin = -10e-6/lengthToSI;
  double ymax = 10e-6/lengthToSI;
  double zmin = 0;//-17e-6/lengthToSI;
  double zmax = 17e-6/lengthToSI;

  long double Etot=0;
  unsigned long long int ntot=0;
  long double wsum=0;

#pragma omp parallel
{
  // Implicitly private variables declared outside of the loop
  char filename[256];
  float partf[numvars];
  double part[numvars];
  float u2,ek,thet;
  int ebin, abin, j;
  size_t i;
  unsigned long int counter;
  FILE *data;
#pragma omp for schedule(guided) reduction(+:hist, Etot, ntot, wsum)
  for(i=0;i<count;i++){
      // Check if the filename is one we want
      //if (!strncmp(filelist[i], "boundary", 8)) continue;

      sprintf(filename, "%s", globbuf.gl_pathv[i]);
      //fprintf(stderr, "Working on file %s\n", filename);
      counter=0;
      data = fopen(filename, "rb");
      if (data == NULL) ERROR(("Cannot open file %s\n", filename));

      while(1){
          if (fread(partf, sizeof(float), numvars, data) != numvars){
              //printf("fread failed !!!! \n\nEOF was not set!!!!\n\n");
          }
          if (feof(data)){
              //printf("breaking\n");
              break;
          }
          // Cast to double precission to avoid "wierd" floting point edge cases
          for (j=0;j<numvars;j++) part[j] = partf[j];

          //fprintf(stdout, "Pos is %g %g %g %g\n", part[x], part[y], part[z], part[ux]);
          // If not in box, ignore
          //if (part[x]<xmin || part[x]>xmax || part[z]<zmin || part[z]>zmax) continue;

          //fprintf(stderr, "starting counter %d\n", counter);
          // Get energy index
          u2 = part[ux]*part[ux] + part[uy]*part[uy] + part[uz]*part[uz];
          // I could save a multiply in this innermost loop by doing all the
          // binning in normalized units, but the performance is dominated by
          // disk access, so let's keep things a bit more understandable
          ek = ekMeVconst * u2/ (1. + sqrt(1.+ u2));
          if (ek < Ecut) continue;
          Etot += ek*part[w]; // Increment Etot before putting it in the hist
          if (ek > emax) ek = emax-0.5*de;
          ebin = (int)(ek/de);
          // Get angle index
          thet = acosf(part[ux]/sqrtf(u2));
          abin = (int)(thet/da);
          // Especially in single precision, the argument of acos above can be
          // negative unity, necessitating this edge case.
          if (abin==nbinsa) abin--;
          if(ebin >= nbinse || abin >= nbinsa || abin < 0){
            fprintf(stderr, "ebin is %d abin is %d count is %d\n", ebin, abin, counter);
            fprintf(stderr, "ux is %.18e uy is %.18e uz is %.18e\n", part[ux], part[uy], part[uz]);
            fprintf(stderr, "The  ux is %.18e and the fraction  is %.18e\n", part[ux], part[ux]/sqrtf(u2));
            fprintf(stderr, "is equal evaluates to %d\n", -1.*part[ux]==sqrtf(u2));
            fprintf(stderr, "ux/sqrtf(u2) is %.18e\nsqrtf(u2)/ux is %.18e\n", sqrtf(u2)/part[ux], part[ux]/sqrtf(u2));
            ntot++;
            continue;
            //ERROR(("You're probably about to segfault"));
          }
          hist[abin][ebin] += part[w];
          wsum += part[w];
          ntot++;
          counter++;
          //fprintf(stderr, "Done with this one\n");
      }
      fclose(data);
      fprintf(stderr, "File %zu had %lu entries\n", i, counter);
  }
}

  count = globbuf.gl_pathc;
  globfree(&globbuf);

  fprintf(stdout, "The total number of simulation particles used is %lld\n", ntot);
  fprintf(stdout, "The total number of physical particles, assuming you didn't do anything weird in your deck, is %Lg\n", wsum);
  fprintf(stdout, "The total energy in the particles is %Lg MeV, or %Lg J.\n", Etot, Etot*1e6*e_SI);

  // Normalize the angle bins
  // The normalization is 4*pi steradians / the solid angle subtended by the bin
  // Let the compiler handle any parallelization here.
  int i,j;
  double norm;
  for(i=0;i<nbinsa;i++){
      norm = 2./((cos(da*i)-cos(da*(i+1)))*de);
      for(j=0;j<nbinse;j++)
          hist[i][j] *= norm;
  }

  // Write the hist params for the Python plotter to read
  FILE * out;
  sprintf(temp, "%s%s", particle, "anglehistparams.txt");
  out = fopen(temp, "w");
  fprintf(out, "# Parameter file used for the Python 2D hist plotter.\n");
  fprintf(out, "%.14e   Angle minimum.\n", amin);
  fprintf(out, "%.14e   Angle maximum\n", amax);
  fprintf(out, "%.14e   Energy minimum.\n", emin);
  fprintf(out, "%.14e   Energy maximum\n", emax);
  fprintf(out, "%d   Number of bins in angle\n", nbinsa);
  fprintf(out, "%d   Number of bins in energy\n", nbinse);
  fprintf(out, "%s   Particle species\n", particle);
  fprintf(out, "%.14e   ekMeVconst\n", ekMeVconst);
  fprintf(out, "%lld   Number of particles used\n", ntot);
  fprintf(out, "%.14e   Total Energy in histogram (MeV)\n", Etot);
  fprintf(out, "%lld   Number of physical particles in histogram\n", wsum);
  fclose(out);

  // Store the histogram for Python plotting
  // Cast to double so numpy can understand
  double histD[nbinsa][nbinse];
  for(i=0;i<nbinsa;i++)
    for(j=0;j<nbinse;j++)
      histD[i][j] = hist[i][j];
  sprintf(temp, "%s%s", particle, "lostspec.bin");
  out = fopen(temp, "w");
  for(i=0;i<nbinsa;i++)
    fwrite(histD[i], sizeof(double), nbinse, out);
  fclose(out);

  return 0; 
} // main


//---------------------------------------------------------------------
// For ERROR macro 
//
void print_log( const char *fmt, ... ) {
 va_list ap;
 va_start( ap, fmt );
 vfprintf( stderr, fmt, ap );
 va_end( ap );
 fflush( stderr );
} // print_log 

