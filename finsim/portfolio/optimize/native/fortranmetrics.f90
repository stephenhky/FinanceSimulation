module fortranmetrics
    implicit none
    private
    public f90_sharpe_ratio, f90_mpt_costfunction, f90_mpt_entropy_costfunction

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


    function f90_mpt_costfunction(nbsymbols, weights, r, cov, rf, lamb, V0) result(mpt_costfunction)
        integer, intent(in) :: nbsymbols
        real, dimension(nbsymbols+1), intent(in) :: weights
        real, dimension(nbsymbols), intent(in) :: r
        real, dimension(nbsymbols, nbsymbols), intent(in) :: cov
        real, intent(in) :: rf
        real, intent(in) :: lamb
        real, intent(in) :: V0
        real :: mpt_costfunction
        real :: c, cashyield, stockyield, volatility
        integer :: i, j

        c = lamb * V0
        cashyield = weights(nbsymbols+1) * rf

        stockyield = 0.
        do i=1, nbsymbols
            stockyield = stockyield + weights(i) * r(i)
        end do

        volatility = 0.
        do i=1, nbsymbols
            do j=1, nbsymbols
                volatility = volatility + weights(i) * cov(i, j) * weights(j)
            end do
        end do

        mpt_costfunction = cashyield + stockyield - 0.5 * c / V0 * volatility

    end function f90_mpt_costfunction


    function f90_mpt_entropy_costfunction(nbsymbols, weights, r, cov, rf, lamb0, lamb1, V) result(mpt_entropy_costfunction)
        integer, intent(in) :: nbsymbols
        real, dimension(nbsymbols+1), intent(in) :: weights
        real, dimension(nbsymbols), intent(in) :: r
        real, dimension(nbsymbols, nbsymbols), intent(in) :: cov
        real, intent(in) :: rf
        real, intent(in) :: lamb0, lamb1
        real, intent(in) :: V
        real :: mpt_entropy_costfunction
        real :: c0, c1, sumweights, cashyield, stockyield, volatility, entropy
        integer :: i, j

        c0 = lamb0 * V
        c1 = lamb1 * V
        cashyield = weights(nbsymbols+1) * rf

        stockyield = 0.
        do i=1, nbsymbols
            stockyield = stockyield + weights(i) * r(i)
        end do

        volatility = 0.
        do i=1, nbsymbols
            do j=1, nbsymbols
                volatility = volatility + weights(i) * cov(i, j) * weights(j)
            end do
        end do

        sumweights = 0.
        do i=1, nbsymbols
            sumweights = sumweights + weights(i)
        end do

        entropy = 0.
        do i=1, nbsymbols
            if (weights(i) > 0) then
                entropy = entropy + weights(i) * (log(weights(i)) - log(sumweights))
            end if
        end do
        entropy = entropy / sumweights

        mpt_entropy_costfunction = cashyield + stockyield - 0.5 * c0 / V * volatility - 0.5 * c1 / V * entropy

    end function f90_mpt_entropy_costfunction


end module fortranmetrics

! compiling
! > f2py --overwrite-signature -h fortranmetrics.pyf -m fortranmetrics fortranmetrics.f90
! > f2py -c fortranmetrics.pyf fortranmetrics.f90 -m fortranmetrics
