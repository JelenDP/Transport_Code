    program trmoao
    implicit none
    call read_trmoao
    end program trmoao
    
    !==================================================================================================================================!
    subroutine read_trmoao()
    implicit none
    

    integer :: nspin, ncao, nsao
    integer :: iunit, ispin, i, j, ierr
    real(8), allocatable :: trmocao(:,:,:), trmosao(:,:,:)
                        
    iunit = 66
    
    write (*,'(A)',advance='no') 'ncao: '
    read (*,*) ncao
    
    write (*,'(A)',advance='no') 'nsao: '
    read (*,*) nsao

    write (*,'(A)',advance='no') 'nspin: '
    read (*,*) nspin
       
    allocate(trmocao(ncao,nsao,nspin), trmosao(nsao,nsao,nspin))
    
    write(6,*) "Open cc-trmocao"
    open(iunit,file="cc-trmocao",form='unformatted',access='direct',recl=8*nsao*ncao)
    rewind(iunit)
    do ispin = 1, nspin
        read(iunit,rec=ispin) trmocao(:,:,ispin)
    end do 
    if (ierr /= 0) then 
        write(6,*) "ERROR: cc-trmocao read ended with error."
    end if
    close(iunit)

    write(6,*) "Write mocao.dat"
    open(iunit,file="mocao.dat",status='unknown')
    rewind(iunit)
    do i = 1, ncao
      do j = 1, nsao
        write(iunit,'(f25.20)')  trmocao(i,j,1)
      end do
    end do
    close(iunit)

    write(6,*) "Open cc-trmosao"
    open(iunit,file="cc-trmosao",form='unformatted',access='direct',recl=8*nsao*ncao)
    rewind(iunit)
    do ispin = 1, nspin
        read(iunit,rec=ispin) trmosao(:,:,ispin)
    end do 
    if (ierr /= 0) then 
        write(6,*) "ERROR: cc-trmosao read ended with error."
    end if
    close(iunit)

    write(6,*) "Write mosao.dat"
    open(iunit,file="mosao.dat",status='unknown')
    rewind(iunit)
    do i = 1, nsao
      do j = 1, nsao
        write(iunit,'(f25.20)')  trmosao(i,j,1)
      end do
    end do
    close(iunit)

    end subroutine read_trmoao
    !==================================================================================================================================!
    