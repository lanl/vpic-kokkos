
// Utility to parse lost particle data into angle-resolved energy deposition
// data
//
// Usage: 
//     ./edep [species name]
//
//
// icc -fopenmp -o edep edep.c
//
// Requires OpenMP 4.5 (gcc 6.1 or Intel 17.0), or you may run slowrially
//
// Note the energy cutoff Ecut below.
//
// This program must match how the lost particle data are written as defined in
// your deck.  TODO: Update this all to hdf5 or similar, so you don't have to
// know the data layout bit for bit.
//
// polarbinplot.py will make a nice plot for you using the output of this
// program.
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

//---------------------------------------------------------------------
// General purpose memory allocation macro 
#define ALLOCATE(A,LEN,TYPE)                                                  \
  if ( !((A)=(TYPE *)malloc((size_t)(LEN)*sizeof(TYPE))) )                    \
      ERROR(("Cannot allocate."));

// Construct an index for the particle data.  This must match how the data are
// output in the lost particle processor in your VPIC deck.
// Sorry this is so ad hoc at the moment.
#define ux 0
#define uy 1
#define uz 2
#define w 3
#define x 4
#define y 5
#define z 6
#define numvars 7

int main( int argc, char *argv[] ) { 
  fprintf(stderr,"Ham.\n");
  int num_tracers_total, nprocs; 
  int nvar; 
  int itmax; 
  char temp[256];
  char usage_msg[] = 
             "Usage: ./edep [species name]\n\n";

  if ( argc != 2 ) { 
    fprintf( stderr, "%s", usage_msg ); 
    exit(0); 
  } 


  // Read some numbers from params.txt
  char buffer[1024];
  //int interval, nstep_total, i;
  FILE *params;
  params = fopen("../params.txt", "r");
  if (!params) ERROR(("Cannot open params.txt."));

  double timeToSI, lengthToSI, massToSI, chargeToSI;
  fgets(buffer, 1024, params);
  fscanf(params, "%lf %[^\n]\n", &timeToSI, buffer);
  fscanf(params, "%lf %[^\n]\n", &lengthToSI, buffer);
  fscanf(params, "%lf %[^\n]\n", &massToSI, buffer);
  fscanf(params, "%lf %[^\n]\n", &chargeToSI, buffer);
  fclose(params);
  char *particle = argv[1];

  float Ecut = 1.; // MeV
  double tmax = M_PI;
  double tmin = 0;
  double pmax = 2.*M_PI;
  double pmin = 0;

  // Must define these to use in the reduction clause
#define nbinsp 100
#define nbinst 100
  double dp = pmax/(double)nbinsp;
  //double dt = tmax/(double)nbinst;
  // When you have 10^13 particles at 10^8 precision, you might need extended
  // precision for the sums.
  long double hist[nbinst][nbinsp] = {0};

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
  double WekMeVconst = 184.*mp_me*elecekMeVconst;
  if (strcmp(particle, "I2")==0) ekMeVconst = carbekMeVconst;
  else if (strcmp(particle, "proton")==0){
      ekMeVconst = protonekMeVconst;
      particle = "I2";
  }
  else if (strcmp(particle, "W")==0){
      ekMeVconst = WekMeVconst;
      particle = "I2";
  }
  else ekMeVconst = elecekMeVconst;

  //TODO: Race this (untested!) bit of code against glob on a large VPIC run on
  //Lustre

  //// Count how many files there are in the lostparts directory
  //char dirname[256] = "../../lostparts";
  //struct dirent *dp;
  //DIR *dir = opendir(dirname);
  //if(!dir) ERROR("Directory %s not found", dirname);
  //int count=0;
  //while (dp = readdir(dir)) ++count;
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
  fprintf(stderr, "count is %zu\n", count);

  // Consider only particles within a certain box
  // Check if the line that does this is commented below!
  double xmin = 20e-6/lengthToSI;
  double xmax = 80e-6/lengthToSI;
  double ymin = -15e-6/lengthToSI;
  double ymax = 15e-6/lengthToSI;
  double zmin = -15e-6/lengthToSI;
  double zmax = 15e-6/lengthToSI;

  long double Etot=0;
  unsigned long long int ntot=0; // There can be a \emph{lot} of particles.

#pragma omp parallel
{
  // Implicitly private variables declared outside of the loop
  char filename[256];
  float partf[numvars];
  double part[numvars];
  float u2,ek,thet, phi;
  int tbin, pbin, j;
  size_t i;
  unsigned long int counter;
  FILE *data;
#pragma omp for schedule(guided) reduction(+:hist, Etot, ntot)
  for(i=0;i<count;i++){
      // Check if the filename is one we want
      //if (!strncmp(filelist[i], "boundary", 8)) continue;

      sprintf(filename, "%s", globbuf.gl_pathv[i]);
      //fprintf(stderr, "Working on file %s.\n", filename);
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

          // If not in box, ignore
          //if (part[x]<xmin || part[x]>xmax || part[z]<zmin || part[z]>zmax) continue;

          //fprintf(stderr, "starting counter %lu\n", counter);
          // Get energy index
          u2 = part[ux]*part[ux] + part[uy]*part[uy] + part[uz]*part[uz];
          ek = ekMeVconst * u2/ (1. + sqrt(1.+ u2));
          if (ek<Ecut) continue;
          Etot += ek*part[w]; // Increment Etot before putting it in the hist
          phi = atan2f(part[uy], part[uz]) + M_PI;
          pbin = (int)(phi/dp);
          // Get angle index
          thet = acosf(part[ux]/sqrtf(u2));
          //tbin = (int)(thet/dt);
          tbin = (int) nbinst*.5*(1. - cosf(thet));//TODO: cos of acos
          // Especially in single precision, the argument of acos above can be
          // negative unity, necessitating this edge case.
          if (tbin==nbinst) tbin--;
          // Similarly for atan2
          if (pbin==nbinsp) pbin--;
          if(pbin >= nbinsp || tbin >= nbinst || tbin < 0 || pbin < 0){
            fprintf(stderr, "pbin is %d tbin is %d count is %lu\n", pbin, tbin, counter);
            fprintf(stderr, "ux is %.18e uy is %.18e uz is %.18e\n", part[ux], part[uy], part[uz]);
            fprintf(stderr, "The  ux is %.18e and the fraction  is %.18e\n", part[ux], part[ux]/sqrtf(u2));
            fprintf(stderr, "is equal evaluates to %d\n", -1.*part[ux]==sqrtf(u2));
            fprintf(stderr, "ux/sqrtf(u2) is %.18e\nsqrtf(u2)/ux is %.18e\n", sqrtf(u2)/part[ux], part[ux]/sqrtf(u2));
            ERROR(("You're probably about to segfault"));
      }
          hist[tbin][pbin] += part[w]*ek;
          ntot++;
          counter++;
          //fprintf(stderr, "Done with this one\n");
      }
      fclose(data);
      fprintf(stderr, "File %zu had %lu entries\n", i, counter);
  }
}

  globfree(&globbuf);

  fprintf(stdout, "The total number of simulation particles used is %lld\n", ntot);
  fprintf(stdout, "The total energy in the particles is %Lg MeV, or %Lg J.\n", Etot, Etot*1e6*e_SI);

  // Write the hist params for the Python plotter to read
  FILE * out;
  sprintf(temp, "%s%s", particle, "edepparams.txt");
  out = fopen(temp, "w");
  fprintf(out, "# Parameter file used for the Python 2D hist plotter.\n");
  fprintf(out, "%.14e   Theta minimum.\n", tmin);
  fprintf(out, "%.14e   Theta maximum\n", tmax);
  fprintf(out, "%.14e   Phi minimum.\n", pmin);
  fprintf(out, "%.14e   Phi maximum\n", pmax);
  fprintf(out, "%d   Number of bins in theta\n", nbinst);
  fprintf(out, "%d   Number of bins in phi\n", nbinsp);
  fprintf(out, "%s   Particle species\n", particle);
  fprintf(out, "%.14e   ekMeVconst\n", ekMeVconst);
  fprintf(out, "%llu   Number of particles used\n", ntot);
  fprintf(out, "%.14e   Total Energy in histogram (MeV*weight)\n", Etot);
  fclose(out);

  int i,j;
  // Store the histogram for Python plotting
  // Cast to double so numpy can understand
  double histD[nbinst][nbinsp];
  for(i=0;i<nbinst;i++)
    for(j=0;j<nbinsp;j++)
      histD[i][j] = hist[i][j];
  sprintf(temp, "%s%s", particle, "edep.bin");
  out = fopen(temp, "w");
  for(i=0;i<nbinst;i++)
    fwrite(histD[i], sizeof(double), nbinsp, out);
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

