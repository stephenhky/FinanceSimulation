module fortranfit
    implicit none
    private
    public f90_fit_BlackScholesMerton_model

contains

    function f90_fit_BlackScholesMerton_model(nbpts, ts, prices) result(rsigma_tuple)
        integer, intent(in) :: nbpts
        real, dimension(nbpts), intent(in) :: ts, prices
        real, dimension(2) :: rsigma_tuple
        real :: dlogS, dt
        integer :: i
        real :: sumpr, sumnoise, sumsqnoise

        sumpr = 0.
        sumnoise = 0.
        sumsqnoise = 0.

        do i=1, nbpts-1
            dlogS = log(prices(i+1) / prices(i))
            dt = ts(i+1) - ts(i)

            sumpr = sumpr + dlogS / dt
            sumnoise = sumnoise + dlogS / sqrt(dt)
            sumsqnoise = sumsqnoise + dlogS * dlogS / dt
        end do
        rsigma_tuple(1) = sumpr / (nbpts - 1)
        rsigma_tuple(2) = sqrt(sumsqnoise / (nbpts - 1) - sumnoise * sumnoise / (nbpts - 1) / (nbpts - 1))

    end function f90_fit_BlackScholesMerton_model

end module fortranfit