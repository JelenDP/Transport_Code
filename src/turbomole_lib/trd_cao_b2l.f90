program print_trnd
implicit none
call ricc2trnden
end program print_trnd

!==================================================================================================================================!
subroutine ricc2trnden()
!----------------------------------------------------------------------------------------------------------------------------------!
! Convert TURBOMOLE transition density matrix (trd) binary file to human readble file
! Trd stored in a "complex" way, therefore it might be won't work with other TM version
! Tested for TM version: TURBOMOLE_FDE_30012023
!
! 13.10.2023
! Dávid P. Jelenfi  
!----------------------------------------------------------------------------------------------------------------------------------!
implicit none

character(len=10) method                        !method
character(len=16) ctypden                       !input filename prefix
character(len=16) typout                        !input filename prefix
character(len=20) trd_out                       !name of output file 
character(len=80) trd_in                        !name of binary output from TM, contains trd
character(len=4) ityp                           !label of irrep
integer :: ncao, nspin, nstate, nirrep, dim     !number of: CAOs, spins, states in irrep, irreps, and dimension
integer :: imult, ispin, istate                 !multiplicity, spin, number of state 
integer :: iao, jao, irec                             !index for MOs reading/writing
integer :: iunit, ierr, irrep                   !file iots, some tmp for len, some indexies
character(len=4), allocatable :: ityps(:)       !list of irreps
integer, allocatable :: multis(:)               !list of multiplicities
integer, allocatable :: states(:)               !list of number of states
real(8), allocatable :: sym(:,:), asym(:,:), tmp(:)    !symmetric and anti-symmetric part of trd
real(8)  glb_fac, off_fac, facoff, facdia, res       !factor for global and off-diagonal elements of trd
logical  b_asym, exists                          

!----------------------------------------------------------------------------------------------------------------------------------!
! set parameters      
!----------------------------------------------------------------------------------------------------------------------------------!
iunit = 66

! ask from user
write (*,'(A)',advance='no') 'spin: '
read (*,*) nspin

if (nspin.ne.1) then
  write(6,*) "Only RHF implemeted"
end if

write (*,'(A)',advance='no') 'method: '
read (*,*) method

write (*,'(A)',advance='no') 'ncao: '
read (*,*) ncao

write (*,'(A)',advance='no') 'nirrep: '
read (*,*) nirrep

allocate(ityps(nirrep))
allocate(multis(nirrep))
allocate(states(nirrep))

do irrep=1, nirrep
  write (*,'(a,i2,a)',advance='no') '  the type of ', irrep, ' irrep: '
  read (*,*) ityps(irrep)
  write (*,'(a,i2,a)',advance='no') '  the multiplicity of ', irrep, ' irrep: '
  read (*,*) multis(irrep)
  write (*,'(a,i2,a)',advance='no') '  states in ', irrep, ' irrep: '
  read (*,*) states(irrep)
end do

! hard coded based on TURBOMOLE_FDE_30012023 version
glb_fac = 1.0d0
off_fac = 0.5d0 !implemeted only for this case
b_asym = .true.

if (method == "ccs") then
  ctypden = 'ccs-tmf0'
  typout = 'trd-L'
else if (method == "cc2") then
  ctypden = 'cc2-tmf0'
  typout = 'trd-L'
else if (method == "adcp2" .or. method == "adc(2)") then
  ctypden = 'adcp2-tm0f'
  typout = 'trd-R'
  glb_fac = sqrt(2.0)
else
  write(6,*) method," is not available. Choose from ccs, cc2 or adc(2)."
end if

facoff = glb_fac * off_fac
facdia = glb_fac

!----------------------------------------------------------------------------------------------------------------------------------!
! read and write transition density file
!----------------------------------------------------------------------------------------------------------------------------------!
dim = ncao*(ncao+1)/2

allocate(sym(ncao,ncao))
allocate(asym(ncao,ncao))
!write(6,*) dim
allocate(tmp(dim))
sym = 0.0
asym = 0.0

do irrep=1, nirrep
  nstate = states(irrep)
  imult = multis(irrep)
  ityp = ityps(irrep)

  do istate=1, nstate
    write(trd_in,'(a,"-",i1,a,"-",i3.3,".cao")') trim(ctypden),imult,trim(ityp),istate
    INQUIRE(FILE=trim(trd_in),EXIST=exists)
    if (exists) then
      open(iunit,file=trim(trd_in),form='unformatted',action="read",access='sequential')
      rewind(iunit)
      !read symmetric part
      read(iunit,iostat=ierr) tmp(:)
      if (ierr /= 0) then 
        write(6,*) "ERROR: ", trim(trd_in), " read ended with error."
      end if
      irec = 1
      do iao = 1, ncao
        do jao = 1, iao
          sym(iao,jao) = tmp(irec)
          sym(jao,iao) = tmp(irec)
          !write(6,*) irec, tmp(irec)
          irec = irec + 1 
        end do
      end do
      !read anti-symmetric part
      if (b_asym) then
        irec = 1
        tmp = 999
        read(iunit,iostat=ierr) tmp(:)
        if (ierr /= 0) then 
          write(6,*) "ERROR: ", trim(trd_in), " read ended with error."
        end if
        do iao = 1, ncao
          do jao = 1, iao
            asym(iao,jao) = -1.d0*tmp(irec)
            asym(jao,iao) = tmp(irec)
            !write(6,*) irec, tmp(irec)
            irec = irec + 1 
          end do
        end do
      end if
     
      close(iunit)
    else
      write(6,*) trim(trd_in)," file does'nt exist."
    end if
        
    !write 
    write(trd_out, '(a,"-",i1,a,"-",i3.3,".dat")') trim(typout),imult,trim(ityp),istate
    write(6,'(5a)') 'Convert ',trim(trd_in),' binary to ',trim(trd_out),' file.'
    open(iunit,file=trim(trd_out),status='unknown')
    rewind(iunit)
    do iao = 1, ncao
      do jao = 1, ncao
        !if (iao.ne.jao) then
        !  res = (1/facoff) * (sym(iao,jao) + asym(iao,jao))
        !else if (iao.eq.jao) then
        !  res = (1/facdia) * sym(iao,jao)
        !end if
        !"Off diagonal factor must be 0.5!"
        write(iunit,'(f25.20)')  (1.d0/glb_fac) * (sym(iao,jao) + asym(iao,jao))
      end do
    end do
    close(iunit)

  end do
end do

deallocate(sym)
deallocate(asym)
deallocate(ityps,multis,states)

end subroutine ricc2trnden
!==================================================================================================================================!
