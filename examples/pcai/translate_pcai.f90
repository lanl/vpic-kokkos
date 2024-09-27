!---------------------------------------
! parallel conversion
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
  parameter(nfiles=15)
  parameter(nbands=0)

  integer sizes(3), subsizes(3), starts(3)
  integer fileinfo, ierror, fh(nfiles+2*nbands), filetype, status(MPI_STATUS_SIZE), output_format, continuous, file_per_slice
  integer(kind=MPI_OFFSET_KIND) :: disp, offset

  character*(40), dimension (nfiles+2*nbands) :: fnames
  CHARACTER*(40) cfname

  parameter(master=0)                                                                                      
  parameter(continuous=1)
  parameter(file_per_slice=2)

end module MPI

module topology
implicit none
  integer topology_x,topology_y,topology_z
  real(kind=8) tx,ty,tz
  integer, allocatable :: domain_size(:,:,:,:), idxstart(:,:), idxstop(:,:) 
end module topology

program translate
  use topology
  use MPI
  implicit none
  integer(kind=4)it,itype,ndim,ndomains,decomp,n,nc(3),record_length,ix,iy,iz,yidx, ib, f
  integer(kind=4)nx,ny,nz,nxstart,nxstop,output_record,tindex,nout,i,j,error,yslice,nzstop,nzstart,k, tindex_new, tindex_start
  integer dom_x, dom_y, dom_z
  real(kind=4)time
  real(kind=8) nx_d,ny_d,nz_d,mi_me,dt
  real(kind=8) xmax,ymax,zmax
  real(kind=4), allocatable, dimension(:,:,:) :: ex,ey,ez,bx,by,bz,jx,jy,jz,rho,ne,ux,uy,uz,pxx,pyy,pzz,pxy,pxz,pyz,phi, phit
  real(kind=4), allocatable, dimension(:,:,:) :: rhob,rhof,exc,ezc,buffer,absJ,absB,aniso
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

  type :: ht_type
     
     integer(kind=4) :: tx,ty,tz         ! number of processes in x and y
     integer(kind=4) :: nx,ny,nz         ! number of cells in each direction that belong to this process
     integer(kind=4) :: start_x, stop_x, start_z, stop_z, start_y,stop_y ! where to start/stop in x/y/z
     integer(kind=4) :: ix,iy,iz 

  end type ht_type


! Declare the structures

  type(v0header) :: v0
  type(ht_type) :: ht
  type(fieldstruct), allocatable, dimension(:,:,:) :: field
  type(hydrostruct), allocatable, dimension(:,:,:) :: hydro

! start index

  tindex_start = 0
 
! set output format

  output_format = continuous
  !output_format = file_per_slice

! describe the topology of the conversion program

  ht%tx = 1
  ht%ty = 1
  ht%tz = 1

! init MPI 

  call MPI_INIT(ierr)                                                                                      
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)                                                           
  call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)                                                       

 ! check the topology for consistency

  if ( (ht%tx*ht%ty*ht%tz /= numprocs).or.(topology_x/ht%tx*ht%tx /= topology_x).or.&
       (topology_z/ht%tz*ht%tz /= topology_z).or.(topology_y/ht%ty*ht%ty /= topology_y) ) then

     if (myid == master) print *, "invalid converter topology"
     call MPI_FINALIZE(ierr)
     stop

  endif
  

! read info.bin

! read the info file

  open(unit=10,file='info.bin',form='unformatted', access='stream') !status='old',form='unformatted')
  read(10)tx
  read(10)ty
  read(10)tz
  
  read(10)xmax
  read(10)ymax
  read(10)zmax
  
  read(10)nx_d
  read(10)ny_d
  read(10)nz_d
  
  read(10)dt
  
  close(10)

  mi_me=1

 
  topology_x = floor(tx+0.5)
  topology_y = floor(ty+0.5)
  topology_z = floor(tz+0.5)

  nx = floor(nx_d + 0.5)
  ny = floor(ny_d + 0.5)
  nz = floor(nz_d + 0.5)

  ! check the topology for consistency

  if ( (ht%tx*ht%ty*ht%tz /= numprocs).or.(topology_x/ht%tx*ht%tx /= topology_x).or.&
       (topology_z/ht%tz*ht%tz /= topology_z).or.(topology_y/ht%ty*ht%ty /= topology_y) ) then

     if (myid == master) print *, "invalid converter topology"
     call MPI_FINALIZE(ierr)
     stop

  endif

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
     write(fname,"(A9,I0,A,I0,A)")"fields/T.",tindex,"/fields.",tindex,".0"
     if (tindex .ne. 1) inquire(file=trim(fname),exist=dfile)
  enddo
  nout = tindex-tindex_start

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

  output_record = 1
  tindex = tindex_start

!  output_record = 22
!  tindex = 101514

!   allocate(ex(nx,ny,nz))
!   inquire(iolength=record_length)ex
  

!   inquire(file='data/Ex.gda',exist=dfile)
!   if (dfile) then
!      print *," *** Found pre-existing direct access data files ***"
!      open(unit=101,file='data/Ex.gda',access='direct',recl=record_length,status='unknown',form='unformatted')
!      error = 0
!      do while (error == 0)
!         output_record = output_record + 1
!         read(101,rec=output_record,iostat=error)ex
!      enddo
!      close(101)
!      tindex = (output_record-1)*nout
!      print *," *** Resuming data conversion at output record=",output_record,"   Time index=",tindex
!   else
! !     print *," *** New conversion - no pre-existing gda files"
!      tindex = 0
!      output_record = 1
!   endif

!   deallocate(ex)


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

  !allocate(absJ(ht%nx,ht%ny,ht%nz))
!  allocate(absB(ht%nx,ht%ny,ht%nz))

  allocate(jx(ht%nx,ht%ny,ht%nz))
  allocate(jy(ht%nx,ht%ny,ht%nz))
  allocate(jz(ht%nx,ht%ny,ht%nz))
  allocate(rho(ht%nx,ht%ny,ht%nz))

  allocate(aniso(ht%nx,ht%ny,ht%nz))
  
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

  fnames(1) = 'data/Ex'
  fnames(2) = 'data/Ey'
  fnames(3) = 'data/Ez'
  fnames(4) = 'data/Bx'
  fnames(5) = 'data/By'
  fnames(6) = 'data/Bz'

!  fnames(7) = 'data/Uix'
!  fnames(8) = 'data/Uiy'
!  fnames(9) = 'data/Uiz'
  fnames(10) = 'data/ni'
  fnames(11) = 'data/Uix'
  fnames(12) = 'data/Uiy'
  fnames(13) = 'data/Uiz'
  fnames(14) = 'data/niold'
  fnames(15) = 'data/aniso'
  !fnames(16) = 'data/Pi-xy1'

  !fnames(17) = 'data/Uix0'
  !fnames(18) = 'data/Uiy0'
  !fnames(19) = 'data/Uiz0'
  !fnames(20) = 'data/ni0'
  !fnames(21) = 'data/Pi-xx0'
  !fnames(22) = 'data/Pi-yy0'
  !fnames(23) = 'data/Pi-zz0'
  !fnames(24) = 'data/Pi-yz0'
  !fnames(25) = 'data/Pi-xz0'
  !fnames(26) = 'data/Pi-xy0'
  
  !fnames(27) = 'data/Uex1'
  !fnames(28) = 'data/Uey1'
  !fnames(29) = 'data/Uez1'
  !fnames(30) = 'data/ne1'
  !fnames(31) = 'data/Pe-xx1'
  !fnames(32) = 'data/Pe-yy1'
  !fnames(33) = 'data/Pe-zz1'
  !fnames(34) = 'data/Pe-yz1'
  !fnames(35) = 'data/Pe-xz1'
  !fnames(36) = 'data/Pe-xy1'

!  fnames(37) = 'data/Uex0'
!  fnames(38) = 'data/Uey0'
!  fnames(39) = 'data/Uez0'
!  fnames(40) = 'data/ne0'
!  fnames(41) = 'data/Pe-xx0'
!  fnames(42) = 'data/Pe-yy0'
!  fnames(43) = 'data/Pe-zz0'
!  fnames(44) = 'data/Pe-yz0'
!  fnames(45) = 'data/Pe-xz0'
!  fnames(46) = 'data/Pe-xy0'

  !fnames(27) = 'data/Jx'
  !fnames(28) = 'data/Jy'
  !fnames(29) = 'data/Jz'

  !fnames(30) = 'data/absB'
  !fnames(31) = 'data/absJ'

!!$  do ib = 1,nbands
!!$     write(fnames(nfiles+ib),"(A8,I2.2)")"data/iEB",ib
!!$     write(fnames(nfiles+ib+nbands),"(A8,I2.2)")"data/eEB",ib
!!$  enddo

  if (output_format == continuous) then
     ! open MPI files
     disp = 0
     
     do f=1,nfiles!+2*nbands

        cfname = trim(fnames(f)) //  '.gda'

        call MPI_FILE_OPEN(MPI_COMM_WORLD, cfname, IOR(MPI_MODE_RDWR, MPI_MODE_CREATE), fileinfo, fh(f), ierror)
        call MPI_FILE_SET_VIEW(fh(f),disp, MPI_REAL4, filetype, 'native', MPI_INFO_NULL, ierror)
        
     enddo

  endif


! Loop over time slices


  dfile=.true.
  do while(dfile) 

     if (myid==master) print *, " processing fields; time slice:",tindex

     do dom_x = ht%start_x,ht%stop_x
        do dom_y = ht%start_y,ht%stop_y
           do dom_z = ht%start_z, ht%stop_z

              call index_to_rank(dom_x,dom_y,dom_z,n)

! Read in field data and load into global arrays

              write(fname,"(A9,I0,A,I0,A,I0)")"fields/T.",tindex,"/fields.",tindex,".",n-1

              ! Index 0 does not have proper current, so use index 1 if it exists
              !if (tindex == 0) then
              !write(fname,"(A14,I0,A1,I0)")"fields/fields.",1,".",n-1        
              !   inquire(file=trim(fname1),exist=check)
              !   if (check) fname=fname1
              !endif
              inquire(file=trim(fname),exist=check)
              
              
              if (check) then 
                 open(unit=10,file=trim(fname),status='unknown',form='unformatted',access='stream')
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

              read(10)buffer   ! skip div_e error

              read(10)buffer
              bx(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              by(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              bz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer   ! skip div_b error
              !read(10)buffer   ! skip tca
              !read(10)buffer   ! skip 
              !read(10)buffer   ! skip 
              !read(10)buffer   ! skip rhob


              read(10)buffer
              jx(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              jy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              jz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)

              read(10)buffer
              rho(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)


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

         
              close(10)


!              write(fname,"(A9,I0,A,I0,A,I0)")"fields/T.",tindex,"/fields.",tindex,".",n-1
              write(fname,"(A8,I0,A8,I0,A1,I0)")"hydro/T.",tindex,"/Hhydro.",tindex,".",n-1
              inquire(file=trim(fname),exist=check)
              if (check) then
                 open(unit=10,file=trim(fname),form='unformatted',access='stream')!status='unknown',form='unformatted')
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

              read(10)buffer ! ux
              read(10)buffer ! uy
              read(10)buffer ! uz
              read(10)buffer ! ne
              read(10)buffer ! pxx
              pxx(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
              read(10)buffer ! pyy
              pyy(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                   buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
              read(10)buffer ! pzz
              pzz(idxstart(n,1):idxstop(n,1), idxstart(n,2):idxstop(n,2), idxstart(n,3):idxstop(n,3)) = &
                 buffer(2:nc(1)-1,2:nc(2)-1,2:nc(3)-1)
              
              deallocate(buffer)
           enddo
        enddo
     enddo

   
     
             
     if (output_format == continuous) then
     
        offset = (output_record - 1)*ht%nx*ht%ny*ht%nz
        
     else

        offset = 0

        do f=1,nfiles!+2*nbands

           write(cfname,"(I0)") tindex
           cfname = trim(fnames(f)) // '_' // trim(cfname) // '.gda'

           call MPI_FILE_OPEN(MPI_COMM_WORLD, cfname, IOR(MPI_MODE_RDWR, MPI_MODE_CREATE), fileinfo, fh(f), ierror)
           call MPI_FILE_SET_VIEW(fh(f),disp, MPI_REAL4, filetype, 'native', MPI_INFO_NULL, ierror)
           
        enddo
        
     endif

!     print*,'pxx=',pxx
!     print*,'pyy=',pyy
!     print*,'pzz=',pzz

     aniso=(pyy+pzz)/(2.0*pxx)
!     print*,'aniso=',aniso
     
     call MPI_FILE_WRITE_AT_ALL(fh(1), offset, ex, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(2), offset, ey, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(3), offset, ez, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(4), offset, bx, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(5), offset, by, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(6), offset, bz, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
!     call MPI_FILE_WRITE_AT_ALL(fh(7), offset, jx, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
!     call MPI_FILE_WRITE_AT_ALL(fh(8), offset, jy, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
!     call MPI_FILE_WRITE_AT_ALL(fh(9), offset, jz, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(10), offset, rho, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(11), offset, ux, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(12), offset, uy, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(13), offset, uz, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(14), offset, ne, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)
     call MPI_FILE_WRITE_AT_ALL(fh(15), offset, aniso, ht%nx*ht%ny*ht%nz, MPI_REAL4, status, ierror)


 

     if (output_format == file_per_slice) then
        do f=1,nfiles!+2*nbands
           call MPI_FILE_CLOSE(fh(f), ierror)
        enddo
     endif
     
     ! Check if there is another time slice to read
     
     
     dfile = .false.
     tindex_new=tindex
     do while ((.not.dfile).and.(tindex_new < tindex+nout))        
        tindex_new=tindex_new+1
        if (tindex_new .GT. 1) then
           write(fname,"(A9,I0,A,I0,A)")"fields/T.",tindex_new,"/fields.",tindex_new,".0"
           inquire(file=trim(fname),exist=dfile)
        endif
     enddo
     
     tindex=tindex_new     
     if (dfile) output_record=output_record+1
     
  enddo ! time loop

  if (output_format == continuous) then
     do f=1,nfiles+2*nbands
        call MPI_FILE_CLOSE(fh(f), ierror)
     enddo
  endif


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

