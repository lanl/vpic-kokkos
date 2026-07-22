!---------------------------------------------------------------------------------------
!  This program computes the flux surfaces - Ay, by reading the the inplane magnetic field
!  components from data/bx.gda and data/bz.gda, taking the curl, and solving the needed
!  Poisson equation
!
!  This version is for a single sheet - periodic in x, with conducting BC in z
!---------------------------------------------------------------------------------------

program translate
  implicit none
  integer input_record,output_record,input_error,output_error,record_length,it1,it2,nx,ny,nz,i,j,it
  real(kind=4)diff,xmax,ymax,zmax,time
  real(kind=4), allocatable, dimension(:,:) :: bx,bz,Ay
  real(kind=8) dx,dz
  logical :: file_exists

  it1=1
  it2=1000  ! Keep as upper limit
  output_record = 0
  input_error = 0  ! Initialize error flag

  open(unit=10,file='data/info',status='old',form='unformatted')
  read(10)nx,ny,nz
  read(10)xmax,ymax,zmax 

  dx = xmax/real(nx)
  dz = zmax/real(nz)

  print *,"---------------------------------------------------"
  print *,"xmax=",xmax,"    zmax=",zmax
  print *,"nx=",nx,"   nz=",nz
  print *,"dx=",dx,"   dz=",dz
  print *,"---------------------------------------------------"

  allocate(bx(nx,nz))
  allocate(bz(nx,nz))
  allocate(Ay(nx,nz))

  inquire(iolength=record_length)bx
  print *," Setting record length=",record_length

  open(unit=20,file='data/bx.gda',access='direct',recl=record_length,&
       status='unknown',form='unformatted',action='read')     
  open(unit=30,file='data/bz.gda',access='direct',recl=record_length,&
       status='unknown',form='unformatted',action='read')    

  ! Loop until we hit an error (no more records)
  do input_record = it1,it2

     print *,"Reading record=",input_record

     ! Try to read with error handling
     read(20,rec=input_record,iostat=input_error)bx
     if (input_error /= 0) then
        print *,"Reached end of file at record",input_record
        exit  ! Exit the loop
     endif

     read(30,rec=input_record,iostat=input_error)bz
     if (input_error /= 0) then
        print *,"Error reading bz.gda at record",input_record
        exit
     endif

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

     print *,"Saving Ay"
     output_record = output_record +1
     open(unit=40,file='data/Ay_int.gda',access='direct',recl=record_length,&
          status='unknown',form='unformatted',action='write')     
     write(40,rec=output_record)Ay
     close(40)

  enddo

  print *,"Successfully processed",output_record,"records"

  close(20)
  close(30)

end program translate