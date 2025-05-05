$m_p = w^\top \hat{\mu}$

$s_p = \sqrt{w^\top \hat{\Sigma}w}$

$q_{0.05} = F_{t_v}^{-1}(0.05)$

$f_{t_v}(q_{0.05})$

$CVaR_{0.05} = m_p + s_p \frac{f_{t_v}(q_{0.05}) \dot (\nu + q_{0.05}^2)}{(\nu - 1) \cdot 0.05}$

$CVaR_{0.05} = m_p + s_p \cdot C$

$C = \frac{f_{t_v}(q_{0.05}) \dot (\nu + q_{0.05}^2)}{(\nu - 1) \cdot 0.05}$

$$
\max_{w} \quad m_p = w^\top \hat{\mu} \\
\min_{w} \quad CVaR_{0.05} = m_p + \sqrt{w^\top \hat{\Sigma}w} \cdot C \\
s.t. \quad w > 0 \\
\quad \quad w^\top \mathbf{1} = 1
$$


