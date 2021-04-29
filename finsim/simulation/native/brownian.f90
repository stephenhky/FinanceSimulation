module f90brownian
    implicit none
    private
    public inv_errfcn, normaldistsampling

    real, parameter :: pi = 3.141592653589793
    real, parameter :: erfcoef3 = pi/12.
    real, parameter :: erfcoef5 = 7.*pi*pi/480.
    real, parameter :: erfcoef7 = 127.*pi*pi*pi/40320.
    real, parameter :: erfcoef9 = 4369.*pi*pi*pi*pi/5806080.
    real, parameter :: erfcoef11 = 34807.*pi*pi*pi*pi*pi/182476800.

    integer :: iseed(1)

contains

    function inv_errfcn(u) result(z)
        real, intent(in) :: u
        real :: z
        real :: u2, u3, u5, u7, u9, u11

        u2 = u*u
        u3 = u*u2
        u5 = u3*u2
        u7 = u5*u2
        u9 = u7*u2
        u11 = u9*u2

        z = u + erfcoef3*u3 + erfcoef5*u5 + erfcoef7*u7 + erfcoef9*u9 + erfcoef11*u11
        z = 0.5 * sqrt(pi) * z

    end function inv_errfcn


    function normaldistsampling() result(z)
        real :: z
        real :: uniform_r, u

        call random_number(uniform_r)
        u = 2*uniform_r - 1

        z = inv_errfcn(u)

    end function normaldistsampling



end module f90brownian