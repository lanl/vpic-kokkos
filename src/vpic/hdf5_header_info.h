#ifndef VPIC_HDF5_HEADER_INFO_H_
#define VPIC_HDF5_HEADER_INFO_H_

// XML header stuff
extern const char *header;
extern const char *header_topology;
extern const char *header_geom;
extern const char *header_origin;
extern const char *header_dxdydz;
extern const char *footer_geom;
extern const char *grid_line;
extern const char *grid_line_footer;
extern const char *footer;
extern const char *main_body_head;
extern const char *main_body_foot;
extern const char *main_body_attributeV;
extern const char *main_body_attributeS;

#define create_file_with_header(xml_file_name, dimensions, orignal, dxdydz, nframes, fields_interval) \
  {                                                                                                   \
    FILE *fp;                                                                                         \
    fp = fopen(xml_file_name, "w");                                                                   \
    fputs(header, fp);                                                                                \
    fprintf(fp, header_topology, dimensions);                                                         \
    fputs(header_geom, fp);                                                                           \
    fprintf(fp, header_origin, orignal);                                                              \
    fprintf(fp, header_dxdydz, dxdydz);                                                               \
    fputs(footer_geom, fp);                                                                           \
    fprintf(fp, grid_line, nframes);                                                                  \
    int i;                                                                                            \
    for (i = 0; i < nframes; i++)                                                                     \
      fprintf(fp, "%d ", i*fields_interval);                                                         \
    fputs(grid_line_footer, fp);                                                                      \
    fclose(fp);                                                                                       \
  }
#define write_main_body_attribute(fpp, main_body_attribute_p, attribute_name, dims_4d_p, dims_3d_p, file_name_pre_p, time_step_p, a1, a2, a3) \
  {                                                                                                                                           \
    fprintf(fpp, main_body_attribute_p, attribute_name, dims_4d_p,                                                                            \
            dims_3d_p, time_step_p, file_name_pre_p, time_step_p, time_step_p, a1,                                                            \
            dims_3d_p, time_step_p, file_name_pre_p, time_step_p, time_step_p, a2,                                                            \
            dims_3d_p, time_step_p, file_name_pre_p, time_step_p, time_step_p, a3);                                                           \
  }

#define invert_field_xml_item(xml_file_name, speciesname_p, time_step, dims_4d, dims_3d, add_footer_flag)                                     \
  {                                                                                                                                           \
    FILE *fp;                                                                                                                                 \
    fp = fopen(xml_file_name, "a");                                                                                                           \
    fprintf(fp, main_body_head, time_step);                                                                                                   \
    if (field_dump_flag.enabledE())                                                                                                           \
      write_main_body_attribute(fp, main_body_attributeV, "E", dims_4d, dims_3d, speciesname_p, time_step, "ex", "ey", "ez");                 \
    if (field_dump_flag.flags["div_e_err"])                                                                                                   \
      fprintf(fp, main_body_attributeS, "div_e_err", dims_3d, time_step, speciesname_p, time_step, time_step, "div_e_err");                   \
    if (field_dump_flag.enabledCB())                                                                                                          \
      write_main_body_attribute(fp, main_body_attributeV, "B", dims_4d, dims_3d, speciesname_p, time_step, "cbx", "cby", "cbz");              \
    if (field_dump_flag.enabledCB0())                                                                                                         \
      write_main_body_attribute(fp, main_body_attributeV, "B0", dims_4d, dims_3d, speciesname_p, time_step, "cbx0", "cby0", "cbz0");          \
    if (field_dump_flag.flags["div_b_err"])                                                                                                   \
      fprintf(fp, main_body_attributeS, "div_b_err", dims_3d, time_step, speciesname_p, time_step, time_step, "div_b_err");                   \
    if (field_dump_flag.enabledTCA())                                                                                                         \
      write_main_body_attribute(fp, main_body_attributeV, "TCA", dims_4d, dims_3d, speciesname_p, time_step, "tcax", "tcay", "tcaz");         \
    if (field_dump_flag.flags["rhob"])                                                                                                        \
      fprintf(fp, main_body_attributeS, "rhob", dims_3d, time_step, speciesname_p, time_step, time_step, "rhob");                             \
    if (field_dump_flag.enabledJF())                                                                                                          \
      write_main_body_attribute(fp, main_body_attributeV, "JF", dims_4d, dims_3d, speciesname_p, time_step, "jfx", "jfy", "jfz");             \
    if (field_dump_flag.flags["rhof"])                                                                                                        \
      fprintf(fp, main_body_attributeS, "rhof", dims_3d, time_step, speciesname_p, time_step, time_step, "rhof");                             \
    if (field_dump_flag.enabledJFOLD())                                                                                                       \
      write_main_body_attribute(fp, main_body_attributeV, "JFOLD", dims_4d, dims_3d, speciesname_p, time_step, "jfxold", "jfyold", "jfzold"); \
    if (field_dump_flag.flags["rhofold"])                                                                                                     \
      fprintf(fp, main_body_attributeS, "rhofold", dims_3d, time_step, speciesname_p, time_step, time_step, "rhofold");                       \
    if (field_dump_flag.flags["te0"])                                                                                                       \
      fprintf(fp, main_body_attributeS, "te0", dims_3d, time_step, speciesname_p, time_step, time_step, "te0");                           \
    if (field_dump_flag.enabledT())                                                                                                           \
      write_main_body_attribute(fp, main_body_attributeV, "T", dims_4d, dims_3d, speciesname_p, time_step, "tx", "ty", "tz");                 \
    if (field_dump_flag.flags["te"])                                                                                                          \
      fprintf(fp, main_body_attributeS, "te", dims_3d, time_step, speciesname_p, time_step, time_step, "te");                                 \
    if (field_dump_flag.enabledO())                                                                                                           \
      write_main_body_attribute(fp, main_body_attributeV, "O", dims_4d, dims_3d, speciesname_p, time_step, "ox", "oy", "oz");                 \
    if (field_dump_flag.flags["oe"])                                                                                                          \
      fprintf(fp, main_body_attributeS, "oe", dims_3d, time_step, speciesname_p, time_step, time_step, "oe");                                 \
    fprintf(fp, "%s", main_body_foot);                                                                                                        \
    if (add_footer_flag)                                                                                                                      \
      fputs(footer, fp);                                                                                                                      \
    fclose(fp);                                                                                                                               \
  }

#ifdef VARIABLE_CHARGE
#define invert_hydro_xml_item(xml_file_name, speciesname_p, time_step, dims_4d, dims_3d, add_footer_flag)                          \
  {                                                                                                                                \
    FILE *fp;                                                                                                                      \
    fp = fopen(xml_file_name, "a");                                                                                                \
    fprintf(fp, main_body_head, time_step);                                                                                        \
    if (hydro_dump_flag.enabledJ())                                                                                                \
      write_main_body_attribute(fp, main_body_attributeV, "J", dims_4d, dims_3d, speciesname_p, time_step, "jx", "jy", "jz");      \
    if (hydro_dump_flag.flags["rho"])                                                                                              \
      fprintf(fp, main_body_attributeS, "rho", dims_3d, time_step, speciesname_p, time_step, time_step, "rho");                    \
    if (hydro_dump_flag.enabledP())                                                                                                \
      write_main_body_attribute(fp, main_body_attributeV, "P", dims_4d, dims_3d, speciesname_p, time_step, "px", "py", "pz");      \
    if (hydro_dump_flag.flags["rho_m"])                                                                                            \
      fprintf(fp, main_body_attributeS, "rho_m", dims_3d, time_step, speciesname_p, time_step, time_step, "rho_m");                \
    if (hydro_dump_flag.enabledTD())                                                                                               \
      write_main_body_attribute(fp, main_body_attributeV, "TD", dims_4d, dims_3d, speciesname_p, time_step, "txx", "tyy", "tzz");  \
    if (hydro_dump_flag.enabledTOD())                                                                                              \
      write_main_body_attribute(fp, main_body_attributeV, "TOD", dims_4d, dims_3d, speciesname_p, time_step, "tyz", "tzx", "txy"); \
    if (hydro_dump_flag.flags["qmin"])                                                                                             \
      fprintf(fp, main_body_attributeS, "qmin", dims_3d, time_step, speciesname_p, time_step, time_step, "qmin");                  \
    if (hydro_dump_flag.flags["qmax"])                                                                                             \
      fprintf(fp, main_body_attributeS, "qmax", dims_3d, time_step, speciesname_p, time_step, time_step, "qmax");                  \
    if (hydro_dump_flag.flags["n_q0"])                                                                                             \
      fprintf(fp, main_body_attributeS, "n_q0", dims_3d, time_step, speciesname_p, time_step, time_step, "n_q0");                  \
    if (hydro_dump_flag.flags["n_q1"])      	      	      	      	      	      	      	      	      	      	      	     \
      fprintf(fp, main_body_attributeS, "n_q1", dims_3d, time_step, speciesname_p, time_step, time_step, "n_q1");                  \
    if (hydro_dump_flag.flags["n_q2"])      	      	      	      	      	      	      	      	      	      	      	     \
      fprintf(fp, main_body_attributeS, "n_q2", dims_3d, time_step, speciesname_p, time_step, time_step, "n_q2");                  \
    if (hydro_dump_flag.flags["n_q3"])      	      	      	      	      	      	      	      	      	      	      	     \
      fprintf(fp, main_body_attributeS, "n_q3", dims_3d, time_step, speciesname_p, time_step, time_step, "n_q3");                  \
    if (hydro_dump_flag.flags["n_q4"])      	      	      	      	      	      	      	      	      	      	      	     \
      fprintf(fp, main_body_attributeS, "n_q4", dims_3d, time_step, speciesname_p, time_step, time_step, "n_q4");                  \
    if (hydro_dump_flag.flags["n_q5"])      	      	      	      	      	      	      	      	      	      	      	     \
      fprintf(fp, main_body_attributeS, "n_q5", dims_3d, time_step, speciesname_p, time_step, time_step, "n_q5");                  \
    fprintf(fp, "%s", main_body_foot);                                                                                             \
    if (add_footer_flag)                                                                                                           \
      fputs(footer, fp);                                                                                                           \
    fclose(fp);                                                                                                                    \
  }

#else

#define invert_hydro_xml_item(xml_file_name, speciesname_p, time_step, dims_4d, dims_3d, add_footer_flag)                          \
  {                                                                                                                                \
    FILE *fp;                                                                                                                      \
    fp = fopen(xml_file_name, "a");                                                                                                \
    fprintf(fp, main_body_head, time_step);                                                                                        \
    if (hydro_dump_flag.enabledJ())                                                                                                \
      write_main_body_attribute(fp, main_body_attributeV, "J", dims_4d, dims_3d, speciesname_p, time_step, "jx", "jy", "jz");      \
    if (hydro_dump_flag.flags["rho"])                                                                                              \
      fprintf(fp, main_body_attributeS, "rho", dims_3d, time_step, speciesname_p, time_step, time_step, "rho");                    \
    if (hydro_dump_flag.enabledP())                                                                                                \
      write_main_body_attribute(fp, main_body_attributeV, "P", dims_4d, dims_3d, speciesname_p, time_step, "px", "py", "pz");      \
    if (hydro_dump_flag.flags["rho_m"])                                                                                            \
      fprintf(fp, main_body_attributeS, "rho_m", dims_3d, time_step, speciesname_p, time_step, time_step, "rho_m");                \
    if (hydro_dump_flag.enabledTD())                                                                                               \
      write_main_body_attribute(fp, main_body_attributeV, "TD", dims_4d, dims_3d, speciesname_p, time_step, "txx", "tyy", "tzz");  \
    if (hydro_dump_flag.enabledTOD())                                                                                              \
      write_main_body_attribute(fp, main_body_attributeV, "TOD", dims_4d, dims_3d, speciesname_p, time_step, "tyz", "tzx", "txy"); \
    fprintf(fp, "%s", main_body_foot);                                                                                             \
    if (add_footer_flag)                                                                                                           \
      fputs(footer, fp);                                                                                                           \
    fclose(fp);                                                                                                                    \
  }
#endif // #ifdef VARIABLE_CHARGE

#define invert_fluid_xml_item(xml_file_name, speciesname_p, time_step, dims_4d, dims_3d, add_footer_flag)                          \
  {                                                                                                                                \
    FILE *fp;                                                                                                                      \
    fp = fopen(xml_file_name, "a");                                                                                                \
    fprintf(fp, main_body_head, time_step);                                                                                        \
    fprintf(fp, main_body_attributeS, "den", dims_3d, time_step, speciesname_p, time_step, time_step, "den");                      \
    fprintf(fp, main_body_attributeS, "tmp", dims_3d, time_step, speciesname_p, time_step, time_step, "tmp");                      \
    fprintf(fp, main_body_attributeS, "prs", dims_3d, time_step, speciesname_p, time_step, time_step, "prs");                      \
    write_main_body_attribute(fp, main_body_attributeV, "u", dims_4d, dims_3d, speciesname_p, time_step, "ux", "uy", "uz");        \
    fprintf(fp, "%s", main_body_foot);                                                                                             \
    if (add_footer_flag)                                                                                                           \
      fputs(footer, fp);                                                                                                           \
    fclose(fp);                                                                                                                    \
  }
#endif
