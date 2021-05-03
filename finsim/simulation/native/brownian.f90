module f90brownian
    implicit none
    private
    public symmatd2eigen, inv_errfcn, normaldistsampling, initialize_random_seeds, lognormal_price_simulation
    public squareroot_diffusion_simulation, normal2ddistsampling, heston_price_simulation

    real, parameter :: sqrt2 = sqrt(2.)

contains

    subroutine symmatd2eigen(symmat, eigenvalues, eigenvectors)
        real, intent(in), dimension(2, 2) :: symmat
        real, intent(out), dimension(2) :: eigenvalues
        real, intent(out), dimension(2, 2) :: eigenvectors
        real :: a, b, c, norm1, norm2

        a = symmat(1, 1)
        b = symmat(2, 2)
        c = symmat(1, 2)

        if (c == 0) then
            eigenvalues(1) = a
            eigenvalues(2) = b
            eigenvectors(1, 1) = 1.
            eigenvectors(2, 2) = 1.
            eigenvectors(1, 2) = 0.
            eigenvectors(2, 1) = 0.
        else
            eigenvalues(1) = 0.5*(a+b+sqrt((a+b)*(a+b)-4*(a*b-c*c)))
            eigenvalues(2) = 0.5*(a+b-sqrt((a+b)*(a+b)-4*(a*b-c*c)))

            norm1 = sqrt(c*c+(eigenvalues(1)-a)*(eigenvalues(1)-a))
            norm2 = sqrt(c*c+(eigenvalues(2)-a)*(eigenvalues(2)-a))
            eigenvectors(1, 1) = c / norm1
            eigenvectors(2, 1) = (eigenvalues(1)-a) / norm1
            eigenvectors(1, 2) = c / norm2
            eigenvectors(2, 2) = (eigenvalues(2)-a) / norm2
        end if


    end subroutine symmatd2eigen


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


    function normal2ddistsampling(rho) result(zvec)
        real, intent(in), dimension(2, 2) :: rho
        real, dimension(2) :: zvec
        real :: z1, z2
        real, dimension(2, 2) :: X
        real, dimension(2) :: variances

        z1 = normaldistsampling()
        z2 = normaldistsampling()

        call symmatd2eigen(rho, variances, X)

        zvec(1) = X(1, 1)*z1*sqrt(variances(1)) + X(1, 2)*z2*sqrt(variances(2))
        zvec(2) = X(2, 1)*z1*sqrt(variances(1)) + X(2, 2)*z2*sqrt(variances(2))

    end function normal2ddistsampling


    subroutine initialize_random_seeds()
        integer :: values(1:8), k
        integer, dimension(:), allocatable :: seed

        call date_and_time(values=values)
        call random_seed(size=k)
        allocate(seed(1:k))
        seed(:) = values(8)

    end subroutine initialize_random_seeds


    ! normal distribution: https://en.wikipedia.org/wiki/Normal_distribution
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


    function squareroot_diffusion_simulation(x0, theta, kappa, sigma, dt, nbsteps, nbsimulations) result(xarray)
        real, intent(in) :: x0, theta, kappa, sigma, dt
        integer, intent(in) :: nbsteps, nbsimulations
        real, dimension(nbsimulations, nbsteps) :: xarray
        integer :: i, j
        real :: sqrtdt, z

        sqrtdt = sqrt(dt)

        do i=1, nbsimulations
            xarray(i, 1) = x0
            do j=2, nbsteps
                z = normaldistsampling()
                xarray(i, j) = xarray(i, j-1) + kappa*(theta-xarray(i, j-1))*dt
                xarray(i, j) = xarray(i, j) + sigma*sqrt(xarray(i, j-1))*z*sqrtdt
            end do
        end do

    end function squareroot_diffusion_simulation


    subroutine heston_price_simulation(logS0, r, v0, theta, kappa, sigma_v, rho, dt, nbsteps, nbsimulations, S, v)
        real, intent(in) :: logS0, r, v0, theta, kappa, sigma_v, rho, dt
        integer, intent(in) :: nbsteps, nbsimulations
        real, intent(out), dimension(nbsimulations, nbsteps) :: S, v
        real, dimension(nbsimulations, nbsteps) :: logS
        integer :: i, j
        real, dimension(2) :: z
        real, dimension(2, 2) :: rhomat
        real :: sqrtdt

        sqrtdt = sqrt(dt)

        ! covariance matrix
        rhomat(1, 1) = 1.
        rhomat(2, 2) = 1.
        rhomat(1, 2) = rho
        rhomat(2, 1) = rho

        do i=1, nbsimulations
            v(i, 1) = v0
            logS(i, 1) = logS0
            do j=2, nbsteps
                z = normal2ddistsampling(rhomat)
                v(i, j) = v(i, j-1) + kappa*(theta-v(i, j-1))*dt + sigma_v*sqrt(v(i, j-1))*z(2)*sqrtdt
                logS(i, j) = logS(i, j-1) + (r-0.5*v(i, j)*v(i, j))*dt + v(i, j)*z(1)*sqrtdt
                S(i, j) = exp(logS(i, j))
            end do
        end do

    end subroutine heston_price_simulation


end module f90brownian






! f2py --overwrite-signature -h brownian.pyf -m f90brownian brownian.f90
! f2py -c brownian.pyf brownian.f90 -m f90brownian

