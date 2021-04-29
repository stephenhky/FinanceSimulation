module f90brownian
    implicit none
    private
    public inv_errfcn, normaldistsampling

contains

    function inv_errfcn(x) result(z)
        real, intent(in) :: x
        real :: w, p
        real :: z

        w = -log((1.0-x)*(1.0+x))

        if (w < 5.0) then
            w = w - 2.5
            p = 2.81022636e-08
            p = 3.43273939e-07 + p*w
            p = -3.5233877e-06 + p*w
            p = -4.39150654e-06 + p*w
            p = 0.00021858087 + p*w
            p = -0.00125372503 + p*w
            p = -0.00417768164 + p*w
            p = 0.246640727 + p*w
            p = 1.50140941 + p*w
        else
            w = sqrt(w) - 3.0
            p = -0.000200214257
            p = 0.000100950558 + p*w
            p = 0.00134934322 + p*w
            p = -0.00367342844 + p*w
            p = 0.00573950773 + p*w
            p = -0.0076224613 + p*w
            p = 0.00943887047 + p*w
            p = 1.00167406 + p*w
            p = 2.83297682 + p*w
        end if

        z = p*x

    end function inv_errfcn


    function normaldistsampling() result(z)
        real :: z
        real :: uniform_r, u

        call random_number(uniform_r)
        write(*,*) uniform_r
        u = 2*uniform_r - 1

        z = inv_errfcn(u) * sqrt(2.)

    end function normaldistsampling



end module f90brownian

! reference: https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
! normal distribution: https://en.wikipedia.org/wiki/Normal_distribution
