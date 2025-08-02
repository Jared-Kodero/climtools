module f_stats
  implicit none
contains

  function normcdf(z) result(p)
    real(8), intent(in) :: z
    real(8) :: p
    p = 0.5d0 * (1.0d0 + erf(z / sqrt(2.0d0)))
  end function normcdf


  function autocorr(x, n, lag) result(rho)
    implicit none
    integer, intent(in) :: n, lag
    real(8), intent(in) :: x(n)
    real(8) :: rho
    real(8) :: x_mean, num, den
    integer :: i

    x_mean = sum(x) / dble(n)
    num = 0.0d0
    den = 0.0d0

    do i = 1, n - lag
      num = num + (x(i) - x_mean) * (x(i + lag) - x_mean)
    end do

    do i = 1, n
      den = den + (x(i) - x_mean)**2
    end do

    if (den == 0.0d0) then
      rho = 0.0d0
    else
      rho = num / den
    end if
  end function autocorr

  recursive subroutine quicksort(a, n)
    implicit none
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n
    integer :: i, j
    real(8) :: pivot, temp

    if (n <= 1) return

    pivot = a(n)
    i = 1
    do j = 1, n - 1
      if (a(j) <= pivot) then
        temp = a(i)
        a(i) = a(j)
        a(j) = temp
        i = i + 1
      end if
    end do

    temp = a(i)
    a(i) = a(n)
    a(n) = temp

    call quicksort(a(1:i-1), i-1)
    call quicksort(a(i+1:n), n - i)
  end subroutine quicksort

  subroutine mk_hamed_rao_test(arr, n,alpha, lag, slope, pval, trend, mean, std, tau, z)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: arr(n)
    real(8), intent(out) :: slope, pval, mean, std, tau, z
    integer, intent(out) :: trend 
    real(8), intent(in) :: alpha
    integer, intent(in) :: lag
    integer :: s

    real(8) ::  var_s_mod, n_eff, rho_k, corr_sum
    integer :: i, j, k,  n_slopes
    real(8), allocatable :: slopes(:)
    real(8) :: var_s, intercept

   

    ! Mean and standard deviation
    mean = sum(arr) / dble(n)
    std = sqrt(sum((arr - mean)**2) / dble(n - 1))

    ! Mann-Kendall S
    s = 0
    do i = 1, n - 1
      do j = i + 1, n
        if (arr(j) > arr(i)) then
          s = s + 1
        else if (arr(j) < arr(i)) then
          s = s - 1
        end if
      end do
    end do

    ! Uncorrected variance
    var_s = dble(n) * (n - 1) * (2 * n + 5) / 18.0d0

    ! Autocorrelation correction
    corr_sum = 0.0d0
    do k = 1, lag
      rho_k = autocorr(arr, n, k)
      corr_sum = corr_sum + (1.0d0 - dble(k) / dble(n)) * rho_k
    end do

    ! Effective sample size
    n_eff = dble(n) / (1.0d0 + 2.0d0 * corr_sum)
    if (n_eff < 1.0d0) n_eff = 1.0d0

    ! Modified variance
    var_s_mod = var_s * dble(n) * (n - 1) * (2 * n + 5) / &
                (n_eff * (n_eff - 1.0d0) * (2.0d0 * n_eff + 5.0d0))

    ! Z-statistic
    if (s > 0) then
      z = (s - 1.0d0) / sqrt(var_s_mod)
    else if (s < 0) then
      z = (s + 1.0d0) / sqrt(var_s_mod)
    else
      z = 0.0d0
    end if

    ! p-value
    pval = 2.0d0 * (1.0d0 - normcdf(abs(z)))

    ! Kendall's Tau
    tau = dble(s) / (0.5d0 * dble(n) * (n - 1))

    ! Sen's slope
    n_slopes = n * (n - 1) / 2
    allocate(slopes(n_slopes))
    k = 0
    do i = 1, n - 1
      do j = i + 1, n
        k = k + 1
        slopes(k) = (arr(j) - arr(i)) / dble(j - i)
      end do
    end do
    call quicksort(slopes, n_slopes)
    if (mod(n_slopes, 2) == 1) then
      slope = slopes((n_slopes + 1) / 2)
    else
      slope = 0.5d0 * (slopes(n_slopes / 2) + slopes(n_slopes / 2 + 1))
    end if
    deallocate(slopes)

    ! Theil–Sen intercept
    intercept = mean - slope * (dble(n + 1) / 2.0d0)

    ! Trend decision based on alpha
    if (pval < alpha) then
      if (z > 0.0d0) then
        trend = 1
      else
        trend = -1
      end if
    else
      trend = 0
    end if

  end subroutine mk_hamed_rao_test

end module f_stats





