program energy_spectrum
  implicit none
  character(60) fname
  character(20) tempproc
  character(7) basedir
  character a
  integer tindex,nproc,in,i,j,j1,nbin,ndomains,nx,ny,nz,n
  real(kind=4) gam,ke,rx,wx,wx1,emax,dve,me,z
  logical dfile
  real density,lx,ly,lz,xstart,xstop,delta,x,norm,energy,charge,G,Vb,glab
  integer ix,iy,iz,nxp2,nyp2,nzp2,nppc
  parameter (nbin=500,emax=500.0,me=0.511)
  real(kind=4) fe(nbin),f(nbin)

! Problem  specific variables

  parameter(density = 1.742e19)  !  Electron density part/cc
  parameter(nppc = 6 )  !  computational particles per cell

! Define structures

  type :: particle
     real(kind=4) :: dx, dy, dz    ! Particle position in cell coordinates (on [-1,1])
     integer(kind=4) :: i          !  Index of cell containing the particle
     real(kind=4) :: ux,uy,uz      !  Particle normalized momentum
     real(kind=4) :: q
  end type particle

  type :: header
     integer(kind=4) :: size,ndim,dim
  end type header

  type :: v0header
   integer(kind=4) step,nx,ny,nz
   real(kind=4) dt,dx,dy,dz,x0,y0,z0,cvac,eps0,damp
   integer(kind=4) rank,num_proc
   integer(kind=4) species_id
   real(kind=4) q_m
end type v0header

! Declare stuctures

  type(v0header) :: v0
  type(header) :: h0
  type(particle) :: p

! Combined particle data file

  open(unit=20,file="elc-2.bin",status='unknown',form='binary')

! Problem size

     ndomains = 4
     nx = 128
     ny = 6
     nz = 1

     print *,"Processors domains=",ndomains
     print *,"nx=",nx
     print *,"ny=",ny
     print *,"nz=",nz

! Pick time slice and x region

  tindex = 331760

! Loop over processors

  do nproc = 0,ndomains-1

! Read in header
     write(tempproc,"(I)")nproc
     print *, tempproc
     write(fname,"(A,I6,A,I6,A,A)")trim("particle/T."),tindex,trim("/electron."),tindex,trim("."),adjustl(trim(tempproc))
!     write(fname,"(A,I,A,I)")"particle/tmp/electron.",tindex,".",nproc
     print *,"nproc =",nproc," File=",trim(fname)
     open(unit=10,file=trim(fname),status='unknown',form='binary',action='read')
     call read_boilerplate(10)
     read(10)in
     read(10)in
     read(10)v0
     read(10)h0
     print *,h0%size,h0%ndim,h0%dim
     nxp2 = v0%nx + 2
     nyp2 = v0%ny + 2
     nzp2 = v0%nz + 2
     print *,"nxp2=",nxp2,"nyp2=",nyp2,"nzp2=",nzp2
     print *,"T*Wpe=",(v0%step)*(v0%dt),"  dt=",v0%dt
     print *,"Reading  --> ",trim(fname),"  Rank=",v0%rank

! Loop over particles

     f(:) = 0.0
     do n=1,h0%dim        
        read(10)p
        i = p%i
        iz = i/(nxp2*nyp2)                          
        iy = (i - iz*nxp2*nyp2)/nxp2          
        ix = i - nxp2*(iy+nyp2*iz)                  
        x = v0%x0+((ix-1)+(p%dx+1)*0.5)*v0%dx
        z = v0%z0+((iz-1)+(p%dz+1)*0.5)*v0%dz
        if (mod(n,500000) == 0) print *,x,z
!        if ((abs(z) .lt. 100) .and. mod(n,3) == 0) write(20)x,z,p%ux,p%uy,p%uz
!        if (x > 2000.0 .and. x < 4500.0 .and. abs(z) < 200 .and. mod(n,3) == 0) write(20)x,z,p%ux,p%uy,p%uz
!        if (x < 1200.0 .or. x > 5200.0) exit
        if (abs(z) < 200 .and. mod(n,4) == 0) write(20)x,z,p%ux,p%uy,p%uz

     enddo
     close(10)

  enddo

end program energy_spectrum

subroutine read_boilerplate(nfile)
  implicit none
  integer(kind=1)sizearr(5)
  integer(kind=2)cafevar 
  integer(kind=4)deadbeefvar
  real(kind=4)realone
  real(kind=8)doubleone
  integer nfile
  read(nfile)sizearr
  read(nfile)cafevar
  read(nfile)deadbeefvar
  read(nfile)realone
  read(nfile)doubleone
!  print *, sizearr,cafevar,deadbeefvar,realone,doubleone
  return
end subroutine read_boilerplate

 

!!$subroutine read_boilerplate(nfile)
!!$  implicit none
!!$  integer(kind=1)sizearr(5)
!!$  integer(kind=2)cafevar 
!!$  integer(kind=4)nver,dtype,deadbeefvar
!!$  real(kind=4)realone
!!$  real(kind=8)doubleone
!!$  integer nfile,step,nx,ny,nz
!!$  read(nfile)sizearr
!!$  read(nfile)cafevar
!!$  read(nfile)deadbeefvar
!!$  read(nfile)realone
!!$  read(nfile)doubleone
!!$  read(nfile)nver
!!$  read(nfile)dtype
!!$  read(nfile)step
!!$  read(nfile)nx,ny,nz
!!$  print *, sizearr
!!$  write(*,"(Z4,x,Z,x,f4.1,x,f4.1)")cafevar,deadbeefvar,realone,doubleone
!!$  print *,nver,dtype
!!$  print *,step,nx,ny,nz
!!$  print *,"*** end header ***"
!!$  return
!!$end subroutine read_boilerplate

 
