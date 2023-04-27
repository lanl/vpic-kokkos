!---------------------------------------
! parallel conversion; Using Bill's data type at LANL
! 
! this code convert VPIC output into gda files,
! which are "bricks" of data
! 
!---------------------------------------

module MPI
  include "mpif.h"                                                                                         
  integer myid,numprocs,ierr                                                                               
  integer master  

  ! MPI IO stuff
  integer nfiles, nbands
  parameter(nfiles=36)!66)
  parameter(nbands=0)!0)

  integer sizes(3), subsizes(3), starts(3)
  integer fileinfo, ierror, fh, filetype, status(MPI_STATUS_SIZE), output_format, continuous, file_per_slice
  integer(kind=MPI_OFFSET_KIND) :: disp, offset
  integer err_length,ierror2
  character*(256) err_msg
  character*(40), dimension (nfiles+2*nbands) :: fnames
  CHARACTER*(40) cfname

  parameter(master=0)                                                                                      
  parameter(continuous=1)
  parameter(file_per_slice=2)
  
  integer append_to_files

end module MPI

module topology
implicit none
  integer topology_x,topology_y,topology_z
  real(kind=8) tx,ty,tz
  integer, allocatable :: domain_size(:,:,:,:), idxstart(:,:), idxstop(:,:) 

  type :: ht_type
     
     integer(kind=4) :: tx,ty,tz         ! number of processes in x and y
     integer(kind=4) :: nx,ny,nz         ! number of cells in each direction that belong to this process
     integer(kind=4) :: start_x, stop_x, start_z, stop_z, start_y,stop_y ! where to start/stop in x/y/z
     integer(kind=4) :: ix,iy,iz 

  end type ht_type

  type(ht_type) :: ht

end module topology

program translate
  use topology
  use MPI
  implicit none
  integer(kind=4)it,itype,ndim,ndomains,decomp,n,nc(3),record_length,ix,iy,iz,yidx, ib, f
  integer(kind=4)nx,ny,nz,nxstart,nxstop,output_record,tindex,nskip,nout,i,j,error,yslice,nzstop,nzstart,k, tindex_new, tindex_start,tindex_stop

  integer dom_x, dom_y, dom_z
  integer(kind=4) httx,htty,httz
  real(kind=4)time
  real(kind=8) nx_d,ny_d,nz_d,mi_me,dt
  real(kind=8) xmax,ymax,zmax
  real(kind=4), allocatable, dimension(:,:,:) :: ex,ey,ez,bx,by,bz,jx,jy,jz,rho,ne,ux,uy,uz,pxx,pyy,pzz,pxy,pxz,pyz,phi, phit
  real(kind=4), allocatable, dimension(:,:,:) :: rhob,rhof,exc,ezc,buffer,absJ,absB
  real(kind=4), allocatable, dimension(:,:,:,:) :: eb
  character(40) fname,fname1
  logical dfile,check

! Define structure for V0 header

  type::v0header
     integer(kind=4) :: version, type, nt, nx, ny, nz
     real(kind=4) :: dt, dx, dy, dz
     real(kind=4) :: x0, y0, z0
     real(kind=4) :: cvac, eps0, damp
     integer(kind=4) :: rank, ndom, spid, spqm
  end type v0header

  type :: fieldstruct
     real(kind=4) :: ex, ey, ez, div_e_err         ! Electric field and div E error
     real(kind=4) :: cbx, cby, cbz, div_b_err      ! Magnetic field and div B error
     real(kind=4) :: tcax, tcay, tcaz, rhob        ! TCA fields and bound charge density
     real(kind=4) :: jfx, jfy, jfz, rhof           ! Free current and charge density
     integer(kind=2) :: ematx,ematy, ematz, nmat   ! Material at edge centers and nodes
     integer(kind=2) :: fmatx, fmaty, fmatz, cmat  ! Material at face and cell centers
  end type fieldstruct

  type :: hydrostruct
     real(kind=4) :: jx, jy, jz, rho  ! Current and charge density => <q v_i f>, <q f>
     real(kind=4) :: px, py, pz, ke   ! Momentum and K.E. density  => <p_i f>, <m c^2 (gamma-1) f>
     real(kind=4) :: txx, tyy, tzz    ! Stress diagonal            => <p_i v_j f>, i==j
     real(kind=4) :: tyz, tzx, txy    ! Stress off-diagonal        => <p_i v_j f>, i!=j
     real(kind=4) :: pad1,pad2        ! 16-byte align
  end type hydrostruct

  ! this describes the topology as viewed by the conversion programm



! Declare the structures

  type(v0header) :: v0
  type(fieldstruct), allocatable, dimension(:,:,:) :: field
  type(hydrostruct), allocatable, dimension(:,:,:) :: hydro

! init MPI 

  call MPI_INIT(ierr)                                                                                      
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)                                                           
  call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)                                                       


  namelist /datum/ httx,htty,httz,tindex_start,tindex_stop,output_format,append_to_files 

! read the configuration file

  if (myid==master) then
     open(unit=10,file='conf.dat',form='formatted',status='old')
     read(10,datum)
     close(10)
  endif

  call MPI_BCAST(httx,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)
  call MPI_BCAST(htty,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)
  call MPI_BCAST(httz,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)

  call MPI_BCAST(tindex_start,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)
  call MPI_BCAST(tindex_stop,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)

  call MPI_BCAST(output_format,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)
  call MPI_BCAST(append_to_files,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)
  
  
  ht%tx = httx
  ht%ty = htty
  ht%tz = httz

 ! check the topology for consistency

  if ( (ht%tx*ht%ty*ht%tz /= numprocs).or.(topology_x/ht%tx*ht%tx /= topology_x).or.&
       (topology_z/ht%tz*ht%tz /= topology_z).or.(topology_y/ht%ty*ht%ty /= topology_y) ) then

     if (myid == master) print *, "invalid converter topology"
     call MPI_FINALIZE(ierr)
     stop

  endif
  

! read info.bin

  open(unit=10,file="info.bin",status='unknown',form='binary')

  read(10),tx
  read(10),ty
  read(10),tz
  
  read(10),xmax
  read(10),ymax
  read(10),zmax

  read(10),nx_d
  read(10),ny_d
  read(10),nz_d

  read(10),dt
  read(10),mi_me

!   tx = 6.0
!   ty = 1.0
!   tz = 6.0

!   nx_d = 600.0
!   ny_d = 1.0
!   nz_d = 600.0
  
!   xmax = 120.0
!   ymax = 1.562000e-01
!   zmax = 12.0
  
  ! mi_me = 400.0
  
! convert to integers
 
  topology_x = floor(tx+0.5)
  topology_y = floor(ty+0.5)
  topology_z = floor(tz+0.5)

  nx = floor(nx_d + 0.5)
  ny = floor(ny_d + 0.5)
  nz = floor(nz_d + 0.5)

  ! convert myid to homer indeces

  call rank_to_index(myid,ht%ix,ht%iy,ht%iz,ht%tx,ht%ty,ht%tz)

  ! domain start/stop for this process

  ht%start_x = topology_x/ht%tx*ht%ix  
  ht%stop_x = topology_x/ht%tx*(ht%ix + 1) - 1 

  ht%start_y = topology_y/ht%ty*ht%iy  
  ht%stop_y = topology_y/ht%ty*(ht%iy + 1) - 1 

  ht%start_z = topology_z/ht%tz*ht%iz 
  ht%stop_z = topology_z/ht%tz*(ht%iz + 1) - 1 

  ! numner of cells for each process

  ht%nx = nx/ht%tx
  ht%ny = ny/ht%ty
  ht%nz = nz/ht%tz

  if (myid==master) then

     print *, "-----------------------------------------------"
     print *, " Topology: ", topology_x, topology_y, topology_z
     print *, " nx,nz,nz: ", nx,ny,nz
     print *, " ht: nx,ny,nz: ", ht%nx,ht%ny,ht%nz
     print *, " mass ratio: ", mi_me
     print *, "-----------------------------------------------"
     
  endif

! total number of domains

  ndomains=topology_x*topology_y*topology_z

  allocate(domain_size(topology_x,topology_y,topology_z,3))
  allocate(idxstart(ndomains,3))
  allocate(idxstop(ndomains,3))

! determine total size of global problem

  do n = 1,ndomains
     
     call rank_to_index(n-1,ix,iy,iz,topology_x,topology_y,topology_z)

     domain_size(ix+1,iy+1,iz+1,1) = (nx/topology_x)
     domain_size(ix+1,iy+1,iz+1,2) = (ny/topology_y)
     domain_size(ix+1,iy+1,iz+1,3) = (nz/topology_z)

     idxstart(n,1) = ( (nx/topology_x))*ix+1 - ht%nx*ht%ix
     idxstart(n,2) = ( (ny/topology_y))*iy+1 - ht%ny*ht%iy
     idxstart(n,3) = ( (nz/topology_z))*iz+1 - ht%nz*ht%iz

     idxstop(n,1)  = idxstart(n,1) +  (nx/topology_x) - 1
     idxstop(n,2)  = idxstart(n,2) +  (ny/topology_y) - 1 
     idxstop(n,3)  = idxstart(n,3) +  (nz/topology_z) - 1
     
  enddo

! Determine number of iterations between output files

if (myid == master) then

  dfile=.false.
  tindex= tindex_start
  do while(.not.dfile)
     tindex=tindex+1
     write(fname,"(A,I0,A,I0,A)")"fields/T.",tindex,"/fields.",tindex,".0"
     if (tindex .ne. 1) inquire(file=trim(fname),exist=dfile)
  enddo
  nskip = 1
  nout = (tindex-tindex_start)*nskip
  !nout = 1 !1time part L.O.  

! Total size of domain

  print *,"---------------------------------------------------"
  print *
  print *,"xmax=",xmax,"   ymax=",ymax,"   zmax=",zmax
  print *
  print *,"Iterations between output=",nout
  print *,"---------------------------------------------------"

endif

call MPI_BCAST(nout,1,MPI_INTEGER,master,MPI_COMM_WORLD,ierr)
  
! Need to determine the last record written,so we know which time slice to process next

  if (append_to_files==1) then
   output_record = (tindex_start/nout) + 1
  else
    output_record = 1
  endif

  tindex = tindex_start

! Allocate storage space for fields and moments

  allocate(ex(ht%nx,ht%ny,ht%nz))
  allocate(ey(ht%nx,ht%ny,ht%nz))
  allocate(ez(ht%nx,ht%ny,ht%nz))
  allocate(bx(ht%nx,ht%ny,ht%nz))
  allocate(by(ht%nx,ht%ny,ht%nz))
  allocate(bz(ht%nx,ht%ny,ht%nz))

  allocate(ux(ht%nx,ht%ny,ht%nz))
  allocate(uy(ht%nx,ht%ny,ht%nz))
  allocate(uz(ht%nx,ht%ny,ht%nz))
  allocate(ne(ht%nx,ht%ny,ht%nz))
  allocate(pxx(ht%nx,ht%ny,ht%nz))
  allocate(pyy(ht%nx,ht%ny,ht%nz))
  allocate(pzz(ht%nx,ht%ny,ht%nz))
  allocate(pyz(ht%nx,ht%ny,ht%nz))
  allocate(pxz(ht%nx,ht%ny,ht%nz))
  allocate(pxy(ht%nx,ht%ny,ht%nz))

  allocate(absJ(ht%nx,ht%ny,ht%nz))
  allocate(absB(ht%nx,ht%ny,ht%nz))

  allocate(jx(ht%nx,ht%ny,ht%nz))
  allocate(jy(ht%nx,ht%ny,ht%nz))
  allocate(jz(ht%nx,ht%ny,ht%nz))

  if (nbands > 0)  allocate(eb(nbands,ht%nx,ht%ny,ht%nz))

  if (myid==master) then
          
     ! Write information file for IDL viewer

     open(unit=17,file='data/info',status='replace',form='unformatted')
     write(17)nx,ny,nz
     write(17)real(xmax,kind=4),real(ymax,kind=4),real(zmax,kind=4)
     close(17)
     
  endif  ! end of open files on master


! creat view, open MPI file, etc


  ! size of the global matrix

  sizes(1) = nx
  sizes(2) = ny
  sizes(3) = nz

  ! size of the chunck seen by each process

  subsizes(1) = ht%nx
  subsizes(2) = ht%ny
  subsizes(3) = ht%nz

  ! where each chunck starts

  starts(1) = ht%ix*ht%nx
  starts(2) = ht%iy*ht%ny
  starts(3) = ht%iz*ht%nz

  call MPI_TYPE_CREATE_SUBARRAY(3,sizes,subsizes,starts, MPI_ORDER_FORTRAN, MPI_REAL4, filetype, ierror)
  call MPI_TYPE_COMMIT(filetype, ierror)
  call MPI_INFO_CREATE(fileinfo,ierror)

  call MPI_INFO_SET(fileinfo,"romio_cb_write","enable",ierror)
  call MPI_INFO_SET(fileinfo,"romio_ds_write","disable",ierror)

  fnames(1) = 'data/ex'
  fnames(2) = 'data/ey'
  fnames(3) = 'data/ez'
  fnames(4) = 'data/bx'
  fnames(5) = 'data/by'
  fnames(6) = 'data/bz'

  fnames(7) = 'data/uix'
  fnames(8) = 'data/uiy'
  fnames(9) = 'data/uiz'
  fnames(10) = 'data/ni'
  fnames(11) = 'data/pi-xx'
  fnames(12) = 'data/pi-yy'
  fnames(13) = 'data/pi-zz'
  fnames(14) = 'data/pi-yz'
  fnames(15) = 'data/pi-xz'
  fnames(16) = 'data/pi-xy'

  fnames(17) = 'data/ui2x'
  fnames(18) = 'data/ui2y'
  fnames(19) = 'data/ui2z'
  fnames(20) = 'data/ni2'
  fnames(21) = 'data/pi2-xx'
  fnames(22) = 'data/pi2-yy'
  fnames(23) = 'data/pi2-zz'
  fnames(24) = 'data/pi2-yz'
  fnames(25) = 'data/pi2-xz'
  fnames(26) = 'data/pi2-xy'

  fnames(27) = 'data/ui2x'
  fnames(28) = 'data/ui2y'
  fnames(29) = 'data/ui2z'
  fnames(30) = 'data/ni2'
  fnames(31) = 'data/pi2-xx'
  fnames(32) = 'data/pi2-yy'
  fnames(33) = 'data/pi2-zz'
  fnames(34) = 'data/pi2-yz'
  fnames(35) = 'data/pi2-xz'
  fnames(36) = 'data/pi2-xy'

  !fnames(27) = 'data/jx'
  !fnames(28) = 'data/jy'
  !fnames(29) = 'data/jz'

  !fnames(30) = 'data/absB'
  !fnames(31) = 'data/absJ'



  do ib = 1,nbands
     write(fnames(nfiles+ib),"(A8,I2.2)")"data/P",ib
  !   write(fnames(nfiles+ib+nbands),"(A8,I2.2)")"data/eEB",ib
  enddo

! Loop over time slices

  dfile=.true.
  do while(dfile)

     if (myid==master) print *, " processing fields; time slice:",tindex

     do dom_x = ht%start_x,ht%stop_x
        do dom_y = ht%start_y,ht%stop_y
           do dom_z = ht%start_z, ht%stop_z

              call index_to_rank(dom_x,dom_y,dom_z,n)

! Read in field data and load into global arrays

              write(fname,"(A,I0,A8,I0,A1,I0)")"fields/T.",tindex,"/fields.",tindex,".",n-1        

              ! Index 0 does not have proper current, so use index 1 if it exists
              if (tindex == 0) then
                 write(fname1,"(A18,I0,A1,I0)")"fields/T.1/fields.",1,".",n-1        
                 inquire(file=trim(fname1),exist=check)
                 if (check) fname=fname1
              endif
              inquire(file=trim(fname),exist=check)
              
              
              
              if (check) then 
                 open(unit=10,file=trim(fname),status='unknown',form='binary')
              else
                 print *,"Can't find file:",fname
                 print *
                 print *," ***  Terminating ***"
                 stop
              endif
              call read_boilerplate(10)
              read(10)v0
              read(10)itype
              read(10)ndim
              read(10)nc
              allocate(buffer(nc(1),nc(2),nc(3)))     
              
              read(10)buffer
              ex(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                   buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
              
              read(10)buffer
              ey( idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3) ) = &
                   buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
              
              read(10)buffer
              ez(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                   buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer    !skip div_e error

              read(10)buffer
              bx(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              by(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              bz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
         

              close(10)


! for H ion hydro

              write(fname,"(A,I0,A,I0,A,I0)")"hydro/T.",tindex,"/ihydro.",tindex,".",n-1        
              if (check) then 
                 open(unit=10,file=trim(fname),status='unknown',form='binary')
              else
                 print *,"Can't find file:",fname
                 print *
                 print *," ***  Terminating ***"
                 stop
              endif
              call read_boilerplate(10)
              read(10)v0
              read(10)itype
              read(10)ndim
              read(10)nc

              read(10)buffer
              ux(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              uy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              uz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              ne(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              !read(10)buffer !skip momentum 
              !read(10)buffer 
              !read(10)buffer 
              !read(10)buffer !skip KE
              
              read(10)buffer
              pxx(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pyy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pzz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pyz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pxz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pxy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              do ib=1,nbands
                 read(10)buffer
                 eb(ib,idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
              enddo

              close(10)

              deallocate(buffer)
              
           enddo ! x loop
        enddo ! domain_y loop
     enddo ! domain_z loop
     

     jx = ux
     jy = uy
     jz = uz
     
     where (abs(ne) > 0.0) 
        pxx = (pxx - mi_me*ux*ux/ne)
        pyy = (pyy - mi_me*uy*uy/ne)
        pzz = (pzz - mi_me*uz*uz/ne)
        pxy = (pxy - mi_me*ux*uy/ne)
        pxz = (pxz - mi_me*ux*uz/ne)
        pyz = (pyz - mi_me*uy*uz/ne)
        ux = (ux/ne)
        uy = (uy/ne)
        uz = (uz/ne)
     elsewhere
        pxx = 0.0
        pyy = 0.0
        pzz = 0.0
        pxy = 0.0
        pxz = 0.0
        pyz = 0.0
        ux = 0.0
        uy = 0.0
        uz = 0.0
     endwhere
     
     call write_data(fnames(1),ex,tindex,output_record)
     call write_data(fnames(2),ey,tindex,output_record)
     call write_data(fnames(3),ez,tindex,output_record)

     call write_data(fnames(4),bx,tindex,output_record)
     call write_data(fnames(5),by,tindex,output_record)
     call write_data(fnames(6),bz,tindex,output_record)

     call write_data(fnames(7),ux,tindex,output_record)
     call write_data(fnames(8),uy,tindex,output_record)
     call write_data(fnames(9),uz,tindex,output_record)

     call write_data(fnames(10),ne,tindex,output_record)

     call write_data(fnames(11),pxx,tindex,output_record)
     call write_data(fnames(12),pyy,tindex,output_record)
     call write_data(fnames(13),pzz,tindex,output_record)
     call write_data(fnames(14),pyz,tindex,output_record)
     call write_data(fnames(15),pxz,tindex,output_record)
     call write_data(fnames(16),pxy,tindex,output_record)

! Repeat for debris

     do dom_x = ht%start_x,ht%stop_x
        do dom_y = ht%start_y,ht%stop_y
           do dom_z = ht%start_z, ht%stop_z

              call index_to_rank(dom_x,dom_y,dom_z,n)


              write(fname,"(A,I0,A,I0,A,I0)")"hydro/T.",tindex,"/dihydro.",tindex,".",n-1        
              if (check) then 
                 open(unit=10,file=trim(fname),status='unknown',form='binary')
              else
                 print *,"Can't find file:",fname
                 print *
                 print *," ***  Terminating ***"
                 stop
              endif
              call read_boilerplate(10)
              read(10)v0
              read(10)itype
              read(10)ndim
              read(10)nc
              allocate(buffer(nc(1),nc(2),nc(3)))     


              read(10)buffer
              ux(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              uy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              uz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              ne(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              !read(10)buffer !skip momentum 
              !read(10)buffer 
              !read(10)buffer 
              !read(10)buffer !skip KE
              
              read(10)buffer
              pxx(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pyy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pzz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pyz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pxz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              pxy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              do ib=1,nbands
                 read(10)buffer
                 eb(ib,idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
              enddo

              close(10)

              deallocate(buffer)
              
           enddo ! x loop
        enddo ! domain_y loop
     enddo ! domain_z loop
     

     jx = ux
     jy = uy
     jz = uz
     
     where (abs(ne) > 0.0) 
        pxx = (pxx - mi_me*ux*ux/ne)
        pyy = (pyy - mi_me*uy*uy/ne)
        pzz = (pzz - mi_me*uz*uz/ne)
        pxy = (pxy - mi_me*ux*uy/ne)
        pxz = (pxz - mi_me*ux*uz/ne)
        pyz = (pyz - mi_me*uy*uz/ne)
        ux = (ux/ne)
        uy = (uy/ne)
        uz = (uz/ne)
     elsewhere
        pxx = 0.0
        pyy = 0.0
        pzz = 0.0
        pxy = 0.0
        pxz = 0.0
        pyz = 0.0
        ux = 0.0
        uy = 0.0
        uz = 0.0
     endwhere

     call write_data(fnames(17),ux,tindex,output_record)
     call write_data(fnames(18),uy,tindex,output_record)
     call write_data(fnames(19),uz,tindex,output_record)

     call write_data(fnames(20),ne,tindex,output_record)

     call write_data(fnames(21),pxx,tindex,output_record)
     call write_data(fnames(22),pyy,tindex,output_record)
     call write_data(fnames(23),pzz,tindex,output_record)
     call write_data(fnames(24),pyz,tindex,output_record)
     call write_data(fnames(25),pxz,tindex,output_record)
     call write_data(fnames(26),pxy,tindex,output_record)
                   
     !absB = sqrt(bx**2+by**2+bz**2)
     !call write_data(fnames(30),absB,tindex,output_record)
     
 !    do ib=1,nbands
 !       call write_data(fnames(nfiles+ib),reshape(eb(ib,:,:,:),SHAPE(ne)),tindex,output_record)
 !    enddo

     do ib=1,nbands
!        call write_data(fnames(nfiles+nbands+ib),reshape(eb(ib,:,:,:),SHAPE(ne)),tindex,output_record)
        call write_data(fnames(nfiles+ib),reshape(eb(ib,:,:,:),SHAPE(ne)),tindex,output_record)
     enddo
     
     ! might as well just wait here
     
     call MPI_BARRIER(MPI_COMM_WORLD, ierror)
     
     ! Check if there is another time slice to read
     
     
     dfile = .false.
     tindex_new=tindex+nout
     if ((tindex_new .GT. 1) .and. (tindex_new<=tindex_stop)) then
        write(fname,"(A,I0,A,I0,A)")"fields/T.",tindex_new,"/fields.",tindex_new,".0"
        inquire(file=trim(fname),exist=dfile)
     endif

!     tindex_new=tindex
!     do while ((.not.dfile).and.(tindex_new < tindex+nout).and.(tindex_new<=tindex_stop))        
!        tindex_new=tindex_new+nskip!1
!        if (tindex_new .GT. 1) then
!           write(fname,"(A9,I0,A,I0,A)")"fields/T.",tindex_new,"/fields.",tindex_new,".0"
!           inquire(file=trim(fname),exist=dfile)
!        endif
!     enddo
     
     tindex=tindex_new     
     if (dfile) output_record=output_record+1
     
  enddo ! time loop

  call MPI_FINALIZE(ierr)
  
end program translate

subroutine read_boilerplate(nfile)
  implicit none
  integer(kind=1)sizearr(5)
  integer(kind=2)cafevar 
  integer(kind=4)deadbeefvar
  real(kind=4)realone
  real(kind=8)doubleone
  integer nfile
  read(10)sizearr
  read(10)cafevar
  read(10)deadbeefvar
  read(10)realone
  read(10)doubleone
!  print *, sizearr,cafevar, deadbeefvar, realone, doubleone
  return
end subroutine read_boilerplate

!

subroutine rank_to_index(rank,ix,iy,iz,topology_x,topology_y,topology_z) 
implicit none
integer iix, iiy, iiz, rank,ix,iy,iz,topology_x,topology_y,topology_z

iix  = rank
iiy  = iix/topology_x
iix  = iix - iiy*topology_x
iiz  = iiy/topology_y 
iiy  = iiy - iiz*topology_y

ix = iix
iy = iiy
iz = iiz

end subroutine rank_to_index


subroutine index_to_rank(ix,iy,iz,rank)
use topology
implicit none
integer ix,iy,iz,rank, iix,iiy,iiz

iix = mod(ix,topology_x)
iiy = mod(iy,topology_y)
iiz = mod(iz,topology_z)

!  Compute the rank
rank = iix + topology_x*( iiy + topology_y*iiz ) + 1 

end subroutine index_to_rank




subroutine write_data(fname,data,tindex,output_record)
use topology
use mpi
implicit none
integer tindex,output_record
character *(*) fname
real(kind=4) data(ht%nx,ht%ny,ht%nz)
real(kind=8) mp_elapsed


mp_elapsed = MPI_WTIME()

if (output_format == 1) then
   
   offset = (output_record - 1)*ht%nx*ht%ny*ht%nz
   cfname = trim(fname) // '.gda'
   call MPI_FILE_OPEN(MPI_COMM_WORLD, cfname, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh, ierror)
   if (ierror /= 0 ) then
      call MPI_Error_string(ierror, err_msg, err_length,ierror2)
      print *, "Error in MPI_FILE_OPEN:",trim(err_msg)
   endif
   call MPI_FILE_SET_VIEW(fh,disp, MPI_REAL4, filetype, 'native', MPI_INFO_NULL, ierror)
   if (ierror /= 0 ) then
      call MPI_Error_string(ierror, err_msg, err_length,ierror2)
      print *, "Error in MPI_FILE_SET_VIEW:",trim(err_msg)
   endif

else
   
   offset = 0
   write(cfname,"(I0)") tindex
   cfname = trim(fname) // '_' // trim(cfname) // '.gda'
   call MPI_FILE_OPEN(MPI_COMM_WORLD, cfname, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh, ierror)
   if (ierror /= 0 ) then
      call MPI_Error_string(ierror, err_msg, err_length,ierror2)
      print *, "Error in MPI_FILE_OPEN:",trim(err_msg)
   endif
   call MPI_FILE_SET_VIEW(fh,disp, MPI_REAL4, filetype, 'native', MPI_INFO_NULL, ierror)
   if (ierror /= 0 ) then
      call MPI_Error_string(ierror, err_msg, err_length,ierror2)
      print *, "Error in MPI_FILE_SET_VIEW:",trim(err_msg)
   endif
   
endif

if (myid==master) print *, "writing data to file ",trim(cfname)

call MPI_FILE_WRITE_AT_ALL(fh, offset, data, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)

if (ierror /= 0 ) then
  call MPI_Error_string(ierror, err_msg, err_length,ierror2)
  print *, "Error in MPI_FILE_WRITE:",trim(err_msg)
endif

call MPI_FILE_CLOSE(fh,ierror)

mp_elapsed = MPI_WTIME() - mp_elapsed

if (myid==master) write(*,'(A, F5.1)') " => time(s):",mp_elapsed

end subroutine write_data
