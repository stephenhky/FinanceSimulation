module f90brownian
    implicit none
    private
    public inv_errfcn, normaldistsampling, initialize_random_seeds, lognormal_price_simulation

    real, parameter :: sqrt2 = sqrt(2.)

contains

    ! reference: https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
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
        u = 2*uniform_r - 1

        z = inv_errfcn(u) * sqrt2

    end function normaldistsampling


    ! normal distribution: https://en.wikipedia.org/wiki/Normal_distribution
    function initialize_random_seeds() result(k)
        integer :: values(1:8), k
        integer, dimension(:), allocatable :: seed

        call date_and_time(values=values)
        call random_seed(size=k)
        allocate(seed(1:k))
        seed(:) = values(8)

    end function initialize_random_seeds


    function lognormal_price_simulation(logS0, r, sigma, dt, nbsteps, nbsimulations) result(S)
        real, intent(in) :: logS0, r, sigma, dt
        integer, intent(in) :: nbsteps, nbsimulations
        real, dimension(nbsimulations, nbsteps) :: logS, S
        integer :: i, j
        real :: sigmasq, sqrtdt
        real :: z

        sigmasq = sigma*sigma
        sqrtdt = sqrt(dt)

        do i=1, nbsimulations
            logS(i, 1) = logS0
            S(i, 1) = exp(logS0)
            do j=2, nbsteps
                z = normaldistsampling()
                logS(i, j) = logS(i, j-1) + (r-0.5*sigmasq)*dt + sigma*z*sqrtdt
                S(i, j) = exp(logS(i, j))
            end do
        end do


    end function lognormal_price_simulation



end module f90brownian






! f2py --overwrite-signature -h brownian.pyf -m f90brownian brownian.f90
! f2py -c brownian.pyf brownian.f90 -m f90brownian

