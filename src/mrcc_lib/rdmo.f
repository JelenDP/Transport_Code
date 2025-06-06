      program wrtmo
      implicit none
      integer ifile,nbf,i,j,nn
      real*8 itol
      real*8, target, allocatable :: r8heap(:)
      real*8, allocatable :: c(:,:)

      
      write (*,'(A)',advance='no') 'nbf: '
      read (*,'(i6)') nbf
      
      allocate(c(nbf,nbf),r8heap(2*(nbf**2)))
      
      ifile=333
      !itol=1.d-10
      !
      !open(ifile, file="mrcc_mos", status='unknown', form='formatted') 
      !do i=1,nbf
      !  do j=1,nbf
      !    read(ifile,*) c(j,i)
      !  enddo
      !enddo
      !close(ifile)
      !
      !open(ifile, file="MOCOEF", status='unknown', form='unformatted')
      !call wrtmo_f(r8heap,r8heap,c,ifile,itol,nbf,nbf)
      !close(ifile)
      
      c(:,:) = 999.d0
      
      open(ifile, file="MOCOEF", form='unformatted')
      call rtdmx(r8heap,r8heap,c,ifile,nbf,nbf)
      close(ifile)
      
      
      deallocate(c,r8heap)
      end program wrtmo


C***********************************************************************
      subroutine wrtmo_f(r8heap,i4heap,c,ifile,itol,nbf,nbasis)
C***********************************************************************
C Write MO coefficients
C***********************************************************************
      implicit none
      integer ii,jj,nn,i,j,ifile,nbf,nbasis
      real*8 r8heap(*),c(nbf,nbasis),itol
      integer*4 i4heap(*)
C
      nn=0
       do j=1,nbasis
         do i=1,nbf
           if(dabs(c(i,j)).gt.itol) then
            nn=nn+1
            r8heap(nn*2-1)=c(i,j)
            i4heap((nn-1)*4+3)=i
            i4heap((nn-1)*4+4)=j
           endif
         enddo
       enddo
      write(ifile) nn,(r8heap(jj),jj=1,2*nn)
C
      return
      end
C

C***********************************************************************
      subroutine rtdmx(r8heap,i4heap,h,ifile,nb1,nb2)
C***********************************************************************
C Read one-electron quantities
C***********************************************************************
      implicit none
      integer ii,jj,nn,i,j,ifile,nb1,nb2
      real*8 r8heap(*),h(nb1,nb2)
      integer*4 i4heap(*)
C
c      call dfillzero(h,nb1*nb2)

      read(ifile) nn,(r8heap(jj),jj=1,2*nn)
       do ii=1,nn
        i=i4heap((ii-1)*4+3)
        j=i4heap((ii-1)*4+4)
        if(j.le.nb2) h(i,j)=r8heap(ii*2-1)
        write(6,"(2i4,f12.5)") i,j,h(i,j)
       enddo
C
      return
      end
C
