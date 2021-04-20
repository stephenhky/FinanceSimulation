module fortranmetrics
    implicit none
    private
    public f90_sharpe_ratio, l1norm

contains

    function f90_sharpe_ratio(nbsymbols, weights, r, cov, rf) result(sharpe_ratio)
        integer, intent(in) :: nbsymbols
        real, dimension(nbsymbols), intent(in) :: weights, r
        real, dimension(nbsymbols, nbsymbols), intent(in) :: cov
        real, intent(in) :: rf
        real :: sharpe_ratio
        integer :: i, j
        real :: yieldrate, variance

        yieldrate = 0.
        do i=1, nbsymbols
            yieldrate = yieldrate + weights(i) * r(i)
        end do

        variance = 0.
        do i=1, nbsymbols
            do j=1, nbsymbols
                variance = variance + weights(i) * cov(i, j) * weights(j)
            end do
        end do

        sharpe_ratio = (yieldrate - rf) / sqrt(variance)

    end function f90_sharpe_ratio


    function l1norm(length, vec1, vec2) result(l1dist)
        integer, intent(in) :: length
        real, dimension(length), intent(in) :: vec1, vec2
        real :: l1dist
        integer :: i

        l1dist = 0.
        do i=1, length
           l1dist = l1dist + abs(vec1(i)-vec2(i))
        end do
    end function l1norm

end module fortranmetrics

! compiling
! > f2py --overwrite-signature -h fortranmetrics.pyf -m fortranmetrics fortranmetrics.f90
! > f2py -c fortranmetrics.pyf fortranmetrics.f90 -m fortranmetrics
