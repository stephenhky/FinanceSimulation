module fortranrisk
    implicit none
    private
    public fortran_estimate_downside_risk, fortran_estimate_upside_risk

contains

    function fortran_estimate_downside_risk(nbpts, ts, prices, target_return) result(downside_risk)
        integer, intent(in) :: nbpts
        real, dimension(nbpts), intent(in) :: ts, prices
        real, intent(in) :: target_return
        real :: downside_risk
        integer :: i
        real :: dlogS, dt, sum_down_sqrms, rms

        sum_down_sqrms = 0.
        do i=1, nbpts-1
            dlogS = log(prices(i+1) / prices(i))
            dt = ts(i+1) - ts(i)
            rms = dlogS / sqrt(dt)
            if (rms < target_return) then
                sum_down_sqrms = sum_down_sqrms + (target_return-rms) * (target_return-rms)
            end if
        end do

        downside_risk = sqrt(sum_down_sqrms / (nbpts - 1))

    end function fortran_estimate_downside_risk


    function fortran_estimate_upside_risk(nbpts, ts, prices, target_return) result(upside_risk)
        integer, intent(in) :: nbpts
        real, dimension(nbpts), intent(in) :: ts, prices
        real, intent(in) :: target_return
        real :: upside_risk
        integer :: i
        real :: dlogS, dt, sum_up_sqrms, rms

        sum_up_sqrms = 0.
        do i=1, nbpts-1
            dlogS = log(prices(i+1) / prices(i))
            dt = ts(i+1) - ts(i)
            rms = dlogS / sqrt(dt)
            if (rms > target_return) then
                sum_up_sqrms = sum_up_sqrms + (rms-target_return) * (rms-target_return)
            end if
        end do

        upside_risk = sqrt(sum_up_sqrms / (nbpts - 1))

    end function fortran_estimate_upside_risk


end module fortranrisk