#include "hdf5_header_info.h"

// XML header stuff
const char *header = "<?xml version=\"1.0\"?>\n<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n\t<Domain>\n";
const char *header_topology = "\t\t<Topology Dimensions=\"%s\" TopologyType=\"3DCoRectMesh\" name=\"topo\"/>\n";
const char *header_geom = "\t\t<Geometry Type=\"ORIGIN_DXDYDZ\" name=\"geo\">\n";
const char *header_origin = "\t\t\t<!-- Origin --> \n\t\t\t<DataItem Dimensions=\"3\" Format=\"XML\">%s</DataItem>\n";
const char *header_dxdydz = "\t\t\t<!-- DxDyDz --> \n\t\t\t<DataItem Dimensions=\"3\" Format=\"XML\">%s</DataItem>\n";
const char *footer_geom = "\t\t</Geometry>\n";
const char *grid_line = "\t\t<Grid CollectionType=\"Temporal\" GridType=\"Collection\" Name=\"TimeSeries\"> \n \
\t\t\t<Time TimeType=\"HyperSlab\"> \n \
\t\t\t\t<DataItem Dimensions=\"%d\" Format=\"XML\" NumberType=\"Float\">";
const char *grid_line_footer = "</DataItem> \n\
\t\t\t</Time>\n";
const char *footer = "\t\t</Grid>\n\t</Domain>\n</Xdmf>\n";

const char *main_body_head = "\t\t\t<Grid GridType=\"Uniform\" Name=\"T%d\"> \n \
\t\t\t\t<Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>   \n \
\t\t\t\t<Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>  \n";
const char *main_body_foot = "\t\t\t</Grid>\n";

const char *main_body_attributeV = "\
        \t\t\t\t <Attribute AttributeType =\"Vector\" Center=\"Node\" Name=\"%s\">  \n \
            \t\t\t\t\t<DataItem Dimensions=\" %s \" Function=\"JOIN($0, $1, $2)\" ItemType=\"Function\">  \n \
                \t\t\t\t\t\t<DataItem ItemType=\"Uniform\" Dimensions=\" %s \" DataType=\"Float\" Precision=\"4\" Format=\"HDF\"> T.%d/%s_%d.h5:/Timestep_%d/%s </DataItem>  \n \
                \t\t\t\t\t\t<DataItem ItemType=\"Uniform\" Dimensions=\" %s \" DataType=\"Float\" Precision=\"4\" Format=\"HDF\"> T.%d/%s_%d.h5:/Timestep_%d/%s </DataItem>  \n \
                \t\t\t\t\t\t<DataItem ItemType=\"Uniform\" Dimensions=\" %s \" DataType=\"Float\" Precision=\"4\" Format=\"HDF\"> T.%d/%s_%d.h5:/Timestep_%d/%s </DataItem>  \n \
            \t\t\t\t\t</DataItem>  \n \
        \t\t\t\t</Attribute>  \n ";

const char *main_body_attributeS = "\
        \t\t\t\t <Attribute AttributeType =\"Scalar\" Center=\"Node\" Name=\"%s\">  \n \
                \t\t\t\t\t\t<DataItem ItemType=\"Uniform\" Dimensions=\" %s \" DataType=\"Float\" Precision=\"4\" Format=\"HDF\"> T.%d/%s_%d.h5:/Timestep_%d/%s </DataItem>  \n \
        \t\t\t\t</Attribute>  \n ";
