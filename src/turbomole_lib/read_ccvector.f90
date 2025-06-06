program read_ccvector
implicit none
call read_cc
end program read_ccvector
    
!==================================================================================================================================!
subroutine read_cc()
!----------------------------------------------------------------------------------------------------------------------------------!
! read CC binary files 
!----------------------------------------------------------------------------------------------------------------------------------!
implicit none
character(len=80) filename
character(len=16) ctypden                       !binary output from TM, contains transition density matrix 
character(len=20) trd_out                       
character(len=7)  control
character(len=200) line                         !reading line
character(len=100) line2                        !reading line2
character(len=4) ityp                           !label of irrep
character(len=4) str_istate                     !number of states within an irrep
character(len=10) method
integer :: nmo, nspin, nstate, nirrep           !number of MOs, uhf=2/rhf=1, number of states in a irrep, number of irrep
integer :: imult, ispin, istate                 !multiplicity, spin, number of state 
integer :: imo, jmo                             !index for MOs reading/writing
integer :: iunit, ierr, tmplen, i, irrep, max_states        !file iots, some tmp for len, some indexies, nmax number of states
character(len=4), allocatable :: ityps(:)       !list of irreps
integer, allocatable :: multis(:)               !list of multiplicities
integer, allocatable :: states(:)               !list of number of states
real(8), allocatable :: trdenright(:,:,:),trdenleft(:,:,:)    !right and left transition density matrix (MO,MO,spin)
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
control="control"
INQUIRE(FILE=control,EXIST=exists)
if (exists) then
  open(unit=iunit,file=control,action="read")
  !is it rhf or uhf calculation?
  do 
    read(iunit,'(a)',iostat=ierr) line
    if (ierr /= 0) then 
      exit
    end if
    tmplen = len_trim(line)
    do i=1, tmplen
      if (line(i:i) /= ' ') then
        line2 = TRIM(TRIM(adjustl(line2))//line(i:i))
      end if 
    end do
    if (line(1:4) == "$uhf") then
      nspin = 2
    end if
  end do
  close(iunit)
  !------
  open(unit=iunit,file=control,action="read")
  do
    line2 = ""
    read(iunit,'(a)',iostat=ierr) line
    if (ierr /= 0) then 
      exit
    end if

    tmplen = len_trim(line)
    do i=1, tmplen
      if (line(i:i) /= ' ') then
        line2 = TRIM(TRIM(adjustl(line2))//line(i:i))
      end if 
    end do

    !method
    if (line2(1:6) == "adc(2)") then
      method = "adc(2)"
    else if (line2(1:3) == "cc2") then
      method = "cc2"      
    end if

    !read symmetry
    !write(6,*) TRIM(line2)
    !read symmetry
    !if (line2(2:9) == "symmetry") then
    !  write(line2(10:),'(a)') sym
    !end if

    !read number of AOs = number of MOs
    if (line2(1:7) == "nbf(AO)") then
      read(line2(9:),'(i10)') nmo
    end if
    !read irreps
    if (line2(1:5) == "irrep") then
      i = 7
      ityp = ""
      !read label of irrep
      do while ((line2(i:i) /= "m").AND.(line2(i:i) /= "n"))
        ityp = trim(trim(adjustl(ityp))//line2(i:i))
        i = i + 1
      end do
      ityps(nirrep) = ityp
      ityp=""
      !read multiplicity of irrep
      if (line2(i:i) == "m") then
        read(line2(i+13:i+13),'(i10)') imult
        multis(nirrep) = imult
      else
        multis(nirrep) = 1
      end if
      !read number of states
      
      do while (line2(i:i+4) /= "nexc=")
        i = i + 1
      end do
      i = i + 5
      str_istate = ""
      do while ((line2(i:i) /= "m").AND.(line2(i:i) /= "n"))
        str_istate = trim(trim(adjustl(str_istate))//line2(i:i))
        i = i + 1
      end do
      !states(nirrep) = str_istate
      read(str_istate,'(i4)') states(nirrep)
      str_istate = ""
      nirrep = nirrep + 1
    end if

    line2 = ""
  end do
  close(iunit)
else 
  stop "control file does'nt exist."
end if
!write(6,*) multis(:)
!write(6,*) ityps(:)
!write(6,*) states(:)

if ((method.ne."adc(2)").and.(method.ne."cc2")) then
  stop "Use CC2 or ADC(2)!"
end if

!----------------------------------------------------------------------------------------------------------------------------------!
! read CC binary files 
!----------------------------------------------------------------------------------------------------------------------------------!
allocate(trdenright(nmo,nmo,nspin))
trdenright(:,:,:) = 9999

filename = "CCRE0-1--1---1"
INQUIRE(FILE=trim(filename),EXIST=exists)
if (exists) then
    open(iunit,file=trim(filename),form='unformatted',access='sequential')
    do ispin = 1, nspin
        read(iunit) trdenright(:,:,ispin)
    end do
    close(iunit)
    write(trd_out, '("ccre0",i1,a,"-",i3.3,".dat")') imult,trim(ityp),istate
    write(6,'(5a)') 'Convert ',trim(filename),' binary to ',trim(trd_out),' file.'
    open(iunit,file=trim(trd_out),status='unknown')
    rewind(iunit)
    do ispin = 1, nspin
    do imo = 1, nmo
        do jmo = 1, nmo
        write(iunit,'(f25.20)') trdenright(jmo,imo,nspin)
        end do
    end do
    end do
close(iunit)
else
    write(6,*) trim(filename)," file does'nt exist."
end if
deallocate(trdenright)

end subroutine read_cc
!==================================================================================================================================!
