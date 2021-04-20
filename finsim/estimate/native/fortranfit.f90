module fortranfit
    implicit none
    private
    public f90_fit_BlackScholesMerton_model

contains

    function f90_fit_BlackScholesMerton_model(nbpts, ts, prices) result(rsigma_tuple)
        integer, intent(in) :: nbpts
        real, dimension(nbpts), intent(in) :: ts, prices
        real, dimension(2) :: rsigma_tuple
        real, dimension(nbpts-1) :: dlogS, dt
        integer :: i
        real :: sumpr, sumnoise, sumsqnoise

        do i=1, nbpts-1
            dlogS(i) = log(prices(i+1) / prices(i))
            dt(i) = ts(i+1) - ts(i)
        end do

        sumpr = 0.
        sumnoise = 0.
        sumsqnoise = 0.
        do i=1, nbpts-1
            sumpr = sumpr + dlogS(i) / dt(i)
            sumnoise = sumnoise + dlogS(i) / sqrt(dt(i))
            sumsqnoise = sumsqnoise + dlogS(i) * dlogS(i) / dt(i)
        end do
        rsigma_tuple(1) = sumpr / (nbpts - 1)
        rsigma_tuple(2) = sqrt(sumsqnoise / (nbpts - 1) - sumnoise * sumnoise / (nbpts - 1) / (nbpts - 1))

    end function f90_fit_BlackScholesMerton_model

end module fortranfit