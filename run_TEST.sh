#!/bin/bash

##########################################################################
# Modify only these variables!
DECK="diag_flow"
##########################################################################

VERSION="TEST"
APP="${DECK}.Linux"
BDIR="/home/joe/Code_Dev/vpic-kokkos/build_${DECK}_${VERSION}"
DEVEL_DIR="/home/joe/Code_Dev/stable_vpic/vpic-kokkos/build_${DECK}_STABLE"
EN_FILE="energies"
NEIGHBOR_FILE="neighbor_indices.txt"
PLANES_FILE="neighbor_planes.txt"
CONNECTION_FILE="neighbor_connections.txt"

echo
git checkout joe/move_p_zigzag_v2
git branch
echo

echo
echo "Building the ${VERSION} code for the ${DECK} deck..."

if [ ! -d "$BDIR" ]
then
    echo
    echo "Directory not found... Making directory now."
    mkdir "$BDIR"
    echo "Directory made."
    echo
fi

cd $BDIR
pwd
echo
export CXX=`which g++`
echo "CXX COMPILER = ${CXX}"
echo
cmake .. -DACCUMULATE_J_ZIGZAG=ON        \
         -DBUILD_INTERNAL_KOKKOS=ON      \
         -DCMAKE_BUILD_TYPE=Release      \
         -DENABLE_KOKKOS_OPENMP=ON       \
         -DVPIC_DUMP_ENERGIES=ON         \
         -DVPIC_DUMP_NEIGHBORS=ON        \
         -DCMAKE_INSTALL_PREFIX=install
make -j 2

echo
echo "Now compiling the ${VERSION} ${DECK} deck..."
echo
time ./bin/vpic "../sample/${DECK}"

# *******************************************************************************************************************
# Remove auxiliary text files.
# *******************************************************************************************************************

# Remove the energies to get rid of appends
if [ -e "${EN_FILE}.txt" ]
then
    rm "${EN_FILE}" "${EN_FILE}.txt"
fi

# Remove the neighbor indices to get rid of appends
if [ -e "${NEIGHBOR_FILE}" ]
then
    rm "${NEIGHBOR_FILE}"
fi

# Remove the neighbor planes to get rid of appends
if [ -e "${PLANES_FILE}" ]
then
    rm "${PLANES_FILE}"
fi

# Remove the neighbor connections to get rid of appends
if [ -e "${CONNECTION_FILE}" ]
then
    rm "${CONNECTION_FILE}"
fi

# *******************************************************************************************************************
# Run VPIC-Kokkos
# *******************************************************************************************************************

time ./"${APP}" &> "output.${DECK}_${VERSION}"

# *******************************************************************************************************************
# Plot neighbor data
# *******************************************************************************************************************

# Plot histogram if the neighbor indices are present
if [ -e "${NEIGHBOR_FILE}" ]
then
    echo
    echo "Plotting the neighbor indices..."
    time python3 ../plot_neighbor_indices.py $NEIGHBOR_FILE "neighbor_histogram-${DECK}-${VERSION}.png" $DECK $VERSION
    echo
fi

# Plot histogram if the neighbor planes are present
if [ -e "${PLANES_FILE}" ]
then
    echo
    echo "Plotting the neighbor types..."
    time python3 ../plot_neighbor_planes.py $PLANES_FILE "neighbor_types-${DECK}-${VERSION}.png" $DECK $VERSION
    echo
fi

# Plot bar chart if the neighbor connections are present
if [ -e "${CONNECTION_FILE}" ]
then
    echo
    echo "Plotting the neighbor connections..."
    time python3 ../plot_neighbor_connections.py $CONNECTION_FILE "neighbor_connections-${DECK}-${VERSION}.png" $DECK $VERSION
    echo
fi

# Plot energy differences if present
if [ -e "${DEVEL_DIR}/${EN_FILE}.txt" ]
then
    if [ -e "${EN_FILE}.txt" ]
    then
        echo
        echo "Plotting the percent differences..."
        time python3 ../plot_energy_difference.py "${EN_FILE}.txt" "${DEVEL_DIR}/${EN_FILE}.txt" "${DECK}_energy_differences.png" $DECK
        echo
    fi
else
    echo
    echo "${DEVEL_DIR} contains no ${EN_FILE}.txt for comparison."
    echo
fi
