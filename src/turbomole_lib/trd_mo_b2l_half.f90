program print_trnd
implicit none
call ricc2trnden
end program print_trnd

!==================================================================================================================================!
subroutine ricc2trnden()
!----------------------------------------------------------------------------------------------------------------------------------!
! read and write transition densities
!----------------------------------------------------------------------------------------------------------------------------------!
implicit none

!character(len=4), intent(in) :: ityp
!real(8), intent(in) :: vint(nmo,nmo,nspin)
!real(8) ddot
character(len=80) filename
character(len=16) ctypden                       !binary output from TM, contains transition density matrix 
character(len=20) trd_out                       
character(len=7)  control
character(len=200) line                         !reading line
character(len=100) line2                        !reading line2
character(len=4) ityp                           !label of irrep
character(len=4) str_istate                     !number of states within an irrep
character(len=10) method
integer :: d, nmo, nspin, nstate, nirrep           !number of MOs, uhf=2/rhf=1, number of states in a irrep, number of irrep
integer :: imult, ispin, istate                 !multiplicity, spin, number of state 
integer :: kao                                  !index for MOs reading/writing
integer :: iunit, ierr, tmplen, i, irrep, max_states        !file iots, some tmp for len, some indexies, nmax number of states
character(len=4), allocatable :: ityps(:)       !list of irreps
integer, allocatable :: multis(:)               !list of multiplicities
integer, allocatable :: states(:)               !list of number of states
real(8), allocatable :: trdenright_vector(:), trdenleft_vector(:)
logical exists      !if file exist, then it's TRUE otherwise FALSE

iunit = 66

!----------------------------------------------------------------------------------------------------------------------------------!
! state info:
!----------------------------------------------------------------------------------------------------------------------------------!
max_states = 10
allocate(ityps(max_states))
allocate(multis(max_states))
allocate(states(max_states))
nspin = 1
nirrep = 1

!----------------------------------------------------------------------------------------------------------------------------------!
! read nmo, nspin, istate, multis and ityps from control file        
!----------------------------------------------------------------------------------------------------------------------------------!
write (*,'(A)',advance='no') 'method: '
read (*,*) method

write (*,'(A)',advance='no') 'nmo: '
read (*,*) nmo

write (*,'(A)',advance='no') 'nirrep: '
read (*,*) nirrep

do irrep=1, nirrep
  write (*,'(a,i2,a)',advance='no') '  the type of ', irrep, ' irrep: '
  read (*,*) ityps(irrep)
  write (*,'(a,i2,a)',advance='no') '  the multiplicity of ', irrep, ' irrep: '
  read (*,*) multis(irrep)
  write (*,'(a,i2,a)',advance='no') '  states in ', irrep, ' irrep: '
  read (*,*) states(irrep)
end do

if ((method.ne."adc(2)").and.(method.ne."cc2")) then
  stop "Use CC2 or ADC(2)!"
end if

d = (nmo*(nmo+1))/2

!----------------------------------------------------------------------------------------------------------------------------------!
! read and write transition density file
!----------------------------------------------------------------------------------------------------------------------------------!
do irrep=1, nirrep
  nstate = states(irrep)
  imult = multis(irrep)
  ityp = ityps(irrep)
  do istate=1, nstate
    ! read right transition density:
    if (method == "cc2") then
      ctypden = 'cc2-tm0f'
    else if (method == "adc(2)") then
      ctypden = 'adcp2-tm0f'
    end if
    allocate(trdenright_vector(d))
    do i=1,d
      trdenright_vector(i) = 999
    end do
    write(filename,'(a,"-",i1,a,"-",i3.3,".mo")') trim(ctypden),imult,trim(ityp),istate
    INQUIRE(FILE=trim(filename),EXIST=exists)
    if (exists) then
      open(iunit,file=trim(filename),form='unformatted',action="read")
      rewind(iunit)
      read(iunit,iostat=ierr) trdenright_vector(:)
      close(iunit)
      !write right transition density:
      write(trd_out, '("trd-R-",i1,a,"-",i3.3,".dat")') imult,trim(ityp),istate
      write(6,'(5a)') 'Convert ',trim(filename),' binary to ',trim(trd_out),' file.'
      open(iunit,file=trim(trd_out),status='unknown')
      rewind(iunit)
      do kao = 1, d
        write(iunit,'(f25.20)') trdenright_vector(kao)
      end do
    close(iunit)
    else
       write(6,*) trim(filename)," file does'nt exist."
    end if
    deallocate(trdenright_vector)

    ! read left transition density:
    !if (method == "cc2") then
    !  allocate(trdenleft_vector(((nmo*nmo)/2)+nmo/2))
    !  do i=1,(nmo*nmo)/2+nmo/2
    !    trdenleft_vector(i) = 999
    !  end do
    !  ctypden = 'cc2-tmf0'
    !  write(filename,'(a,"-",i1,a,"-",i3.3,".mo")') trim(ctypden),imult,trim(ityp),istate
    !  INQUIRE(FILE=trim(filename),EXIST=exists)
    !  if (exists) then
    !    open(iunit,file=trim(filename),form='unformatted',access='sequential')
    !    rewind(iunit)
    !    read(iunit,iostat=ierr) trdenleft_vector(:)
    !    close(iunit)
    !    !write left transition density:
    !    write(trd_out, '("trd-L-",i1,a,"-",i3.3,".dat")') imult,trim(ityp),istate
    !    write(6,'(5a)') 'Convert ',trim(filename),' binary to ',trim(trd_out),' file.'
    !    open(iunit,file=trim(trd_out),status='unknown')
    !    rewind(iunit)
    !    do kao = 1, ((nmo*nmo)/2)+nmo/2
    !      write(iunit,'(f25.20)') trdenleft_vector(kao)
    !    end do
    !    close(iunit)
    !  else
    !    write(6,*) trim(filename)," file does'nt exist."
    !  end if
    !end if
    !deallocate(trdenleft_vector)

    
  end do
end do
deallocate(ityps,multis,states)

end subroutine ricc2trnden
!==================================================================================================================================!
