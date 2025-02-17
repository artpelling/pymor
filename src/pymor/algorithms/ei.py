# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module contains algorithms for the empirical interpolation of |Operators|.

The main work for generating the necessary interpolation data is handled by
the :func:`ei_greedy` method. The objects returned by this method can be used
to instantiate an |EmpiricalInterpolatedOperator|.

As a convenience, the :func:`interpolate_operators` method allows to perform
the empirical interpolation of the |Operators| of a given model with
a single function call.
"""

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.pod import pod as pod_alg
from pymor.analyticalproblems.functions import EmpiricalInterpolatedFunction, Function
from pymor.core.logger import getLogger
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interface import RemoteObject
from pymor.parallel.manager import RemoteObjectManager
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


def ei_greedy(U, error_norm=None, atol=None, rtol=None, max_interpolation_dofs=None,
              nodal_basis=False, copy=True, pool=dummy_pool):
    """Generate data for empirical interpolation using EI-Greedy algorithm.

    Given a |VectorArray| `U`, this method generates a collateral basis and
    interpolation DOFs for empirical interpolation of the vectors contained in `U`.
    The returned objects can be used to instantiate an |EmpiricalInterpolatedOperator|
    (with `triangular=True`).

    The interpolation data is generated by a greedy search algorithm, where in each
    loop iteration the worst approximated vector in `U` is added to the collateral basis.

    Parameters
    ----------
    U
        A |VectorArray| of vectors to interpolate.
    error_norm
        Norm w.r.t. which to calculate the interpolation error. If `None`, the Euclidean norm
        is used. If `'sup'`, the sup-norm of the dofs is used.
    atol
        Stop the greedy search if the largest approximation error is below this threshold.
    rtol
        Stop the greedy search if the largest relative approximation error is below this threshold.
    max_interpolation_dofs
        Stop the greedy search if the number of interpolation DOF (= dimension of the collateral
        basis) reaches this value.
    nodal_basis
        If `True`, a nodal interpolation basis is constructed. Note that nodal bases are
        not hierarchical. Their construction involves the inversion of the associated
        interpolation matrix, which might lead to decreased numerical accuracy.
    copy
        If `False`, `U` will be modified during executing of the algorithm.
    pool
        If not `None`, the |WorkerPool| to use for parallelization.

    Returns
    -------
    interpolation_dofs
        |NumPy array| of the DOFs at which the vectors are evaluated.
    collateral_basis
        |VectorArray| containing the generated collateral basis.
    data
        Dict containing the following fields:

        :errors:
            Sequence of maximum approximation errors during greedy search.
        :triangularity_errors:
            Sequence of maximum absolute values of interpolation
            matrix coefficients in the upper triangle (should
            be near zero).
        :coefficients:
            |NumPy array| of coefficients such that `collateral_basis`
            is given by `U.lincomb(coefficients)`.
        :interpolation_matrix:
            The interpolation matrix, i.e., the evaluation of
            `collateral_basis` at `interpolation_dofs`.
    """
    assert not isinstance(error_norm, str) or error_norm == 'sup'
    if pool:  # dispatch to parallel implementation
        assert isinstance(U, (VectorArray, RemoteObject))
        with RemoteObjectManager() as rom:
            if isinstance(U, VectorArray):
                U = rom.manage(pool.scatter_array(U))
            return _parallel_ei_greedy(U, error_norm=error_norm, atol=atol, rtol=rtol,
                                       max_interpolation_dofs=max_interpolation_dofs, copy=copy, pool=pool)

    assert isinstance(U, VectorArray)
    assert len(U) > 0

    logger = getLogger('pymor.algorithms.ei.ei_greedy')
    logger.info('Generating Interpolation Data ...')

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    collateral_basis = U.empty()
    K = np.eye(len(U), dtype=U[0].dofs([0]).dtype)  # matrix s.t. U = U_initial.lincomb(K.T)
    coefficients = np.zeros((0, len(U)))
    max_errs = []
    triangularity_errs = []

    if copy:
        U = U.copy()

    ERR = U

    errs = ERR.norm() if error_norm is None else ERR.sup_norm() if error_norm == 'sup' else error_norm(ERR)
    max_err_ind = np.argmax(errs)
    initial_max_err = max_err = errs[max_err_ind]

    # main loop
    while True:
        if max_interpolation_dofs is not None and len(interpolation_dofs) >= max_interpolation_dofs:
            logger.info('Maximum number of interpolation DOFs reached. Stopping extension loop.')
            logger.info(f'Final maximum interpolation error with '
                        f'{len(interpolation_dofs)} interpolation DOFs: {max_err}')
            break

        logger.info(f'Maximum interpolation error with '
                    f'{len(interpolation_dofs)} interpolation DOFs: {max_err}')

        if atol is not None and max_err <= atol:
            logger.info('Absolute error tolerance reached! Stopping extension loop.')
            break

        if rtol is not None and max_err / initial_max_err <= rtol:
            logger.info('Relative error tolerance reached! Stopping extension loop.')
            break

        # compute new interpolation dof and collateral basis vector
        new_vec = U[max_err_ind].copy()
        new_dof = new_vec.amax()[0][0]
        if new_dof in interpolation_dofs:
            logger.info(f'DOF {new_dof} selected twice for interpolation! Stopping extension loop.')
            break
        new_dof_value = new_vec.dofs([new_dof])[0, 0]
        if new_dof_value == 0.:
            logger.info(f'DOF {new_dof} selected for interpolation has zero maximum error! Stopping extension loop.')
            break
        new_vec *= 1 / new_dof_value
        interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
        collateral_basis.append(new_vec)
        coefficients = np.vstack([coefficients, K[max_err_ind] / new_dof_value])
        max_errs.append(max_err)

        # update U and ERR
        new_dof_values = U.dofs([new_dof])
        U.axpy(-new_dof_values[0, :], new_vec)
        K -= (K[max_err_ind] / new_dof_value) * new_dof_values.T
        errs = ERR.norm() if error_norm is None else ERR.sup_norm() if error_norm == 'sup' else error_norm(ERR)
        max_err_ind = np.argmax(errs)
        max_err = errs[max_err_ind]

    interpolation_matrix = collateral_basis.dofs(interpolation_dofs)
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix) + 1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    if len(triangularity_errs) > 0:
        logger.info(f'Interpolation matrix is not lower triangular with maximum error of {triangularity_errs[-1]}')

    if nodal_basis:
        logger.info('Building nodal basis.')
        inv_interpolation_matrix = spla.inv(interpolation_matrix)
        collateral_basis = collateral_basis.lincomb(inv_interpolation_matrix)
        coefficients = inv_interpolation_matrix.T @ coefficients
        interpolation_matrix = np.eye(len(collateral_basis))

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs,
            'coefficients': coefficients, 'interpolation_matrix': interpolation_matrix}

    return interpolation_dofs, collateral_basis, data


def deim(U, modes=None, pod=True, atol=None, rtol=None, product=None, pod_options={}):
    """Generate data for empirical interpolation using DEIM algorithm.

    Given a |VectorArray| `U`, this method generates a collateral basis and
    interpolation DOFs for empirical interpolation of the vectors contained in `U`.
    The returned objects can be used to instantiate an |EmpiricalInterpolatedOperator|
    (with `triangular=False`).

    The collateral basis is determined by the first :func:`~pymor.algorithms.pod.pod` modes of `U`.

    Parameters
    ----------
    U
        A |VectorArray| of vectors to interpolate.
    modes
        Dimension of the collateral basis i.e. number of POD modes of the vectors in `U`.
    pod
        If `True`, perform a POD of `U` to obtain the collateral basis. If `False`, `U`
        is used as collateral basis.
    atol
        Absolute POD tolerance.
    rtol
        Relative POD tolerance.
    product
        Inner product |Operator| used for the POD.
    pod_options
        Dictionary of additional options to pass to the :func:`~pymor.algorithms.pod.pod` algorithm.

    Returns
    -------
    interpolation_dofs
        |NumPy array| of the DOFs at which the vectors are interpolated.
    collateral_basis
        |VectorArray| containing the generated collateral basis.
    data
        Dict containing the following fields:

        :svals:
            POD singular values.
    """
    assert isinstance(U, VectorArray)

    logger = getLogger('pymor.algorithms.ei.deim')
    logger.info('Generating Interpolation Data ...')

    data = {}

    if pod:
        collateral_basis, svals = pod_alg(U, modes=modes, atol=atol, rtol=rtol, product=product, **pod_options)
        data['svals'] = svals
    else:
        collateral_basis = U

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    interpolation_matrix = np.zeros((0, 0))

    for i in range(len(collateral_basis)):
        logger.info(f'Choosing interpolation point for basis vector {i}.')

        if len(interpolation_dofs) > 0:
            coefficients = spla.solve(interpolation_matrix,
                                      collateral_basis[i].dofs(interpolation_dofs))
            U_interpolated = collateral_basis[:len(interpolation_dofs)].lincomb(coefficients)
            ERR = collateral_basis[i] - U_interpolated
        else:
            ERR = collateral_basis[i]

        # compute new interpolation dof and collateral basis vector
        new_dof = ERR.amax()[0][0]

        if new_dof in interpolation_dofs:
            logger.info(f'DOF {new_dof} selected twice for interpolation! Stopping extension loop.')
            break

        interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
        interpolation_matrix = collateral_basis[:len(interpolation_dofs)].dofs(interpolation_dofs)

    if len(interpolation_dofs) < len(collateral_basis):
        del collateral_basis[len(interpolation_dofs):len(collateral_basis)]

    logger.info('Finished.')

    return interpolation_dofs, collateral_basis, data


def interpolate_operators(fom, operator_names, parameter_sample, error_norm=None,
                          product=None, atol=None, rtol=None, max_interpolation_dofs=None,
                          pod_options={}, alg='ei_greedy', pool=dummy_pool):
    """Empirical operator interpolation using the EI-Greedy/DEIM algorithm.

    This is a convenience method to facilitate the use of :func:`ei_greedy` or :func:`deim`.
    Given a |Model|, names of |Operators|, and a sample of |Parameters|, first
    the operators are evaluated on the solution snapshots of the model for the
    provided parameters. These evaluations are then used as input for
    :func:`ei_greedy`/:func:`deim`.  Finally the resulting interpolation data is used to
    create |EmpiricalInterpolatedOperators| and a new model with the interpolated
    operators is returned.

    Note that this implementation creates *one* common collateral basis for all specified
    operators, which might not be what you want.

    Parameters
    ----------
    fom
        The |Model| whose |Operators| will be interpolated.
    operator_names
        List of keys in the `operators` dict of the model. The corresponding
        |Operators| will be interpolated.
    parameter_sample
        A list of |Parameters| for which solution snapshots are calculated.
    error_norm
        See :func:`ei_greedy`.
        Has no effect if `alg == 'deim'`.
    product
        Inner product for POD computation in :func:`deim`.
        Has no effect if `alg == 'ei_greedy'`.
    atol
        See :func:`ei_greedy`.
    rtol
        See :func:`ei_greedy`.
    max_interpolation_dofs
        See :func:`ei_greedy`.
    pod_options
        Further options for :func:`~pymor.algorithms.pod.pod` algorithm.
        Has no effect if `alg == 'ei_greedy'`.
    alg
        Either `ei_greedy` or `deim`.
    pool
        If not `None`, the |WorkerPool| to use for parallelization.

    Returns
    -------
    eim
        |Model| with |Operators| given by `operator_names` replaced by
        |EmpiricalInterpolatedOperators|.
    data
        Dict containing the following fields:

        :dofs:
            |NumPy array| of the DOFs at which the |Operators| have to be evaluated.
        :basis:
            |VectorArray| containing the generated collateral basis.

        In addition, `data` contains the fields of the `data` `dict` returned by
        :func:`ei_greedy`/:func:`deim`.
    """
    assert alg in ('ei_greedy', 'deim')
    logger = getLogger('pymor.algorithms.ei.interpolate_operators')
    with RemoteObjectManager() as rom:
        operators = [getattr(fom, operator_name) for operator_name in operator_names]
        with logger.block('Computing operator evaluations on solution snapshots ...'):
            if pool:
                logger.info(f'Using pool of {len(pool)} workers for parallel evaluation')
                evaluations = rom.manage(pool.push(fom.solution_space.empty()))
                pool.map(_interpolate_operators_build_evaluations, parameter_sample,
                         fom=fom, operators=operators, evaluations=evaluations)
            else:
                evaluations = operators[0].range.empty()
                for mu in parameter_sample:
                    U = fom.solve(mu)
                    for op in operators:
                        evaluations.append(op.apply(U, mu=mu))

        if alg == 'ei_greedy':
            with logger.block('Performing EI-Greedy:'):
                dofs, basis, data = ei_greedy(evaluations, error_norm, atol=atol, rtol=rtol,
                                              max_interpolation_dofs=max_interpolation_dofs,
                                              copy=False, pool=pool)
        elif alg == 'deim':
            if alg == 'deim' and pool is not dummy_pool:
                logger.warning('DEIM algorithm not parallel. Collecting operator evaluations.')
                evaluations = pool.apply(_identity, x=evaluations)
                evs = evaluations[0]
                for e in evaluations[1:]:
                    evs.append(e, remove_from_other=True)
                evaluations = evs
            with logger.block('Executing DEIM algorithm:'):
                dofs, basis, data = deim(evaluations, modes=max_interpolation_dofs,
                                         atol=atol, rtol=rtol, pod_options=pod_options, product=product)
        else:
            assert False

    ei_operators = {name: EmpiricalInterpolatedOperator(operator, dofs, basis, triangular=(alg == 'ei_greedy'))
                    for name, operator in zip(operator_names, operators)}
    eim = fom.with_(name=f'{fom.name}_ei', **ei_operators)

    data.update({'dofs': dofs, 'basis': basis})
    return eim, data


def interpolate_function(function, parameter_sample, evaluation_points,
                         atol=None, rtol=None, max_interpolation_dofs=None):
    """Parameter separable approximation of a |Function| using Empirical Interpolation.

    This method computes a parameter separated |LincombFunction| approximating
    the input |Function| using Empirical Interpolation :cite`BMNP04`.
    The actual EI Greedy algorithm is contained in :func:`ei_greedy`. This function
    acts as a convenience wrapper, which computes the training data and
    constructs an :class:`~pymor.analyticalproblems.functions.EmpiricalInterpolatedFunction`
    from the data returned by :func:`ei_greedy`.

    .. note::
        If possible, choose `evaluation_points` identical to the coordinates at which
        the interpolated function is going to be evaluated. Otherwise `function` will
        have to be re-evaluated at all new evaluation points for all |parameter values|
        given by `parameter_sample`.


    Parameters
    ----------
    function
        The function to interpolate.
    parameter_sample
        A list of |Parameters| for which `function` is evaluated to generate the
        training data.
    evaluation_points
        |NumPy array| of coordinates at which `function` should be evaluated to
        generate the training data.
    atol
        See :func:`ei_greedy`.
    rtol
        See :func:`ei_greedy`.
    max_interpolation_dofs
        See :func:`ei_greedy`.

    Returns
    -------
    ei_function
        The :class:`~pymor.analyticalproblems.functions.EmpiricalInterpolatedFunction` giving
        the parameter separable approximation of `function`.
    data
        `dict` of additional data as returned by :func:`ei_greedy`.
    """
    assert isinstance(function, Function)
    assert isinstance(evaluation_points, np.ndarray)
    assert evaluation_points.ndim == 2
    assert evaluation_points.shape[1] == function.dim_domain

    snapshot_data = NumpyVectorSpace.from_numpy(
        np.array([function(evaluation_points, mu=mu) for mu in parameter_sample]).T
    )

    dofs, basis, ei_data = ei_greedy(snapshot_data, error_norm='sup',
                                     atol=atol, rtol=rtol, max_interpolation_dofs=max_interpolation_dofs)

    ei_function = EmpiricalInterpolatedFunction(
        function, evaluation_points[dofs], ei_data['interpolation_matrix'], True,
        parameter_sample, ei_data['coefficients'],
        evaluation_points=evaluation_points, basis_evaluations=basis.to_numpy().T
    )

    return ei_function, ei_data


def _interpolate_operators_build_evaluations(mu, fom=None, operators=None, evaluations=None):
    U = fom.solve(mu)
    for op in operators:
        evaluations.append(op.apply(U, mu=mu))


def _parallel_ei_greedy(U, pool, error_norm=None, atol=None, rtol=None, max_interpolation_dofs=None,
                        nodal_basis=False, copy=True):

    assert isinstance(U, RemoteObject)

    logger = getLogger('pymor.algorithms.ei.ei_greedy')
    logger.info('Generating Interpolation Data ...')
    logger.info(f'Using pool of {len(pool)} workers for parallel greedy search')

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    collateral_basis = pool.apply_only(_parallel_ei_greedy_get_empty, 0, U=U)
    max_errs = []
    triangularity_errs = []

    with pool.push({}) as distributed_data:
        errs, snapshot_counts = zip(
            *pool.apply(_parallel_ei_greedy_initialize, U=U, error_norm=error_norm, copy=copy, data=distributed_data)
        )
        snapshot_count = sum(snapshot_counts)
        cum_snapshot_counts = np.hstack(([0], np.cumsum(snapshot_counts)))
        K = np.eye(snapshot_count)  # matrix s.t. U = U_initial.lincomb(K.T)
        coefficients = np.zeros((0, snapshot_count))
        max_err_ind = np.argmax(errs)
        initial_max_err = max_err = errs[max_err_ind]

        # main loop
        while True:

            if max_interpolation_dofs is not None and len(interpolation_dofs) >= max_interpolation_dofs:
                logger.info('Maximum number of interpolation DOFs reached. Stopping extension loop.')
                logger.info(f'Final maximum interpolation error with '
                            f'{len(interpolation_dofs)} interpolation DOFs: {max_err}')
                break

            logger.info(f'Maximum interpolation error with {len(interpolation_dofs)} interpolation DOFs: {max_err}')

            if atol is not None and max_err <= atol:
                logger.info('Absolute error tolerance reached! Stopping extension loop.')
                break

            if rtol is not None and max_err / initial_max_err <= rtol:
                logger.info('Relative error tolerance reached! Stopping extension loop.')
                break

            # compute new interpolation dof and collateral basis vector
            new_vec, local_ind = pool.apply_only(_parallel_ei_greedy_get_vector, max_err_ind, data=distributed_data)
            new_dof = new_vec.amax()[0][0]
            if new_dof in interpolation_dofs:
                logger.info(f'DOF {new_dof} selected twice for interpolation! Stopping extension loop.')
                break
            new_dof_value = new_vec.dofs([new_dof])[0, 0]
            if new_dof_value == 0.:
                logger.info(f'DOF {new_dof} selected for interpolation has zero maximum error! '
                            f'Stopping extension loop.')
                break
            new_vec *= 1 / new_dof_value
            interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
            collateral_basis.append(new_vec)
            global_max_err_ind = cum_snapshot_counts[max_err_ind] + local_ind
            coefficients = np.vstack([coefficients, K[global_max_err_ind] / new_dof_value])
            max_errs.append(max_err)

            errs, new_dof_values = zip(
                *pool.apply(_parallel_ei_greedy_update, new_vec=new_vec, new_dof=new_dof, data=distributed_data)
            )
            new_dof_values = np.hstack(new_dof_values)
            K -= (K[global_max_err_ind] / new_dof_value) * new_dof_values[:, np.newaxis]
            max_err_ind = np.argmax(errs)
            max_err = errs[max_err_ind]

    interpolation_matrix = collateral_basis.dofs(interpolation_dofs)
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix) + 1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    if len(triangularity_errs) > 0:
        logger.info(f'Interpolation matrix is not lower triangular with maximum error of {triangularity_errs[-1]}')
        logger.info('')

    if nodal_basis:
        logger.info('Building nodal basis.')
        inv_interpolation_matrix = spla.inv(interpolation_matrix)
        collateral_basis = collateral_basis.lincomb(inv_interpolation_matrix)
        coefficients = inv_interpolation_matrix.T @ coefficients
        interpolation_matrix = np.eye(len(collateral_basis))

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs,
            'coefficients': coefficients, 'interpolation_matrix': interpolation_matrix}

    return interpolation_dofs, collateral_basis, data


def _parallel_ei_greedy_get_empty(U=None):
    return U.empty()


def _parallel_ei_greedy_initialize(U=None, error_norm=None, copy=None, data=None):
    if copy:
        U = U.copy()
    data['U'] = U
    data['error_norm'] = error_norm
    errs = U.norm() if error_norm is None else U.sup_norm() if error_norm == 'sup' else error_norm(U)
    data['max_err_ind'] = max_err_ind = np.argmax(errs)
    return errs[max_err_ind], len(U)


def _parallel_ei_greedy_get_vector(data=None):
    return data['U'][data['max_err_ind']].copy(), data['max_err_ind']


def _parallel_ei_greedy_update(new_vec=None, new_dof=None, data=None):
    U = data['U']
    error_norm = data['error_norm']

    new_dof_values = U.dofs([new_dof])[0, :]
    U.axpy(-new_dof_values, new_vec)

    errs = U.norm() if error_norm is None else U.sup_norm() if error_norm == 'sup' else error_norm(U)
    data['max_err_ind'] = max_err_ind = np.argmax(errs)
    return errs[max_err_ind], new_dof_values


def _identity(x):
    return x
