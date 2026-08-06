!---------------------------------------------------------------------------------------
!  This program computes the flux surfaces - Ay, by reading the the inplane magnetic field
!  components from data/bx.gda and data/bz.gda, taking the curl, and solving the needed
!  Poisson equation
!
!  This version is for a single sheet - periodic in x, with conducting BC in z
!---------------------------------------------------------------------------------------

program translate
  implicit none
  integer input_record,output_record,input_error,output_error,record_length,it1,it2,nx,ny,nz,i,j,k,it
  real(kind=4)diff,xmax,ymax,zmax,time
  ! Full 3D fields (r,theta,z) as stored on disk; Ay3 is the theta-replicated output.
  real(kind=4), allocatable, dimension(:,:,:) :: bx3,bz3,Ay3
  real(kind=4), allocatable, dimension(:,:) :: bx,bz,Ay
  real(kind=8) dx,dz
  logical :: file_exists

  it1=1
  it2=1000  ! Keep as upper limit
  output_record = 0
  input_error = 0  ! Initialize error flag

  open(unit=10,file='data/info',status='old',form='unformatted')
  read(10)nx,ny,nz          ! nx=nr, ny=ntheta, nz
  read(10)xmax,ymax,zmax

  dx = xmax/real(nx)
  dz = zmax/real(nz)

  print *,"---------------------------------------------------"
  print *,"xmax=",xmax,"    zmax=",zmax
  print *,"nx=",nx,"   ny(theta)=",ny,"   nz=",nz
  print *,"dx=",dx,"   dz=",dz
  print *,"---------------------------------------------------"

  ! Read/write the FULL 3D field (nx*ny*nz); do the flux integral on the
  ! theta=0 layer (k=1) and replicate it across theta so Ay_int.gda has the
  ! same (r,theta,z) layout the plotting script expects.
  allocate(bx3(nx,ny,nz))
  allocate(bz3(nx,ny,nz))
  allocate(Ay3(nx,ny,nz))
  allocate(bx(nx,nz))
  allocate(bz(nx,nz))
  allocate(Ay(nx,nz))

  inquire(iolength=record_length)bx3
  print *," Setting record length (full 3D)=",record_length

  open(unit=20,file='data/bx.gda',access='direct',recl=record_length,&
       status='unknown',form='unformatted',action='read')
  open(unit=30,file='data/bz.gda',access='direct',recl=record_length,&
       status='unknown',form='unformatted',action='read')

  ! Loop until we hit an error (no more records)
  do input_record = it1,it2

     print *,"Reading record=",input_record

     ! Try to read with error handling
     read(20,rec=input_record,iostat=input_error)bx3
     if (input_error /= 0) then
        print *,"Reached end of file at record",input_record
        exit  ! Exit the loop
     endif

     read(30,rec=input_record,iostat=input_error)bz3
     if (input_error /= 0) then
        print *,"Error reading bz.gda at record",input_record
        exit
     endif

     ! Extract the theta=0 layer (j=1) for the in-plane (r,z) flux integral
     do j=1,nz
        do i=1,nx
           bx(i,j) = bx3(i,1,j)
           bz(i,j) = bz3(i,1,j)
        enddo
     enddo

     ! Compute Ay
     Ay(1,1)=0.0

     do i=2,nx
        Ay(i,1)=-dx*bz(i-1,1)+Ay(i-1,1)
     enddo

     do i=1,nx
        do j=2,nz
           Ay(i,j)=dz*bx(i,j-1)+Ay(i,j-1)
        enddo
     enddo

     ! Replicate the (r,z) flux surface across all theta layers for output
     do j=1,nz
        do k=1,ny
           do i=1,nx
              Ay3(i,k,j) = Ay(i,j)
           enddo
        enddo
     enddo

     print *,"Saving Ay"
     output_record = output_record +1
     open(unit=40,file='data/Ay_int.gda',access='direct',recl=record_length,&
          status='unknown',form='unformatted',action='write')
     write(40,rec=output_record)Ay3
     close(40)

  enddo

  print *,"Successfully processed",output_record,"records"

  close(20)
  close(30)

end program translate