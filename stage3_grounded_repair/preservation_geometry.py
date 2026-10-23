"""QR/SVD diagnostics of zero-preservation linear directions; no fitted heads."""
import torch


def svd(matrix):
    kwargs = {'driver': 'gesvd'} if matrix.is_cuda else {}
    return torch.linalg.svd(matrix, full_matrices=False, **kwargs)


def preservation_geometry(preserve, teacher, targets, teacher_ids, rows_per_teacher, notify=None):
    if (preserve.dtype != torch.float64 or teacher.dtype != torch.float64 or targets.dtype != torch.float64
            or preserve.ndim != 2 or teacher.ndim != 2 or targets.ndim != 2
            or preserve.shape[0] < preserve.shape[1] or teacher.shape[1] != preserve.shape[1]
            or targets.shape[0] != teacher.shape[0] or
            sum(rows_per_teacher) != teacher.shape[0] or len(teacher_ids) != len(rows_per_teacher)
            or not all(torch.isfinite(x).all() for x in (preserve, teacher, targets))):
        raise ValueError('Require finite FP64 tall preservation design and aligned teacher targets')
    notify = notify or (lambda phase: None)
    notify('thin_qr')
    q, r = torch.linalg.qr(preserve, mode='reduced')
    qr_residual = float(torch.linalg.vector_norm(q @ r-preserve)/torch.linalg.vector_norm(preserve).clamp_min(1e-300))
    if qr_residual > 1e-10:
        raise ValueError('FP64 QR residual exceeds the fixed validation tolerance')
    del q
    notify('svd_of_qr_factor')
    u, singular, vh = svd(r)
    svd_residual = float(torch.linalg.vector_norm((u*singular) @ vh-r)/torch.linalg.vector_norm(r).clamp_min(1e-300))
    if svd_residual > 1e-10:
        raise ValueError('FP64 SVD residual exceeds the fixed validation tolerance')
    del u, r
    largest = float(singular[0])
    preserve_energy = preserve.square().sum().clamp_min(1e-300)
    teacher_energy = teacher.square().sum().clamp_min(1e-300)
    target_energy = targets.square().sum().clamp_min(1e-300)
    results = {}
    artifact = {'preservation_singular_values': singular.cpu()}
    for precision, dtype in (('fp64_arithmetic', torch.float64), ('fp32_feature_resolution', torch.float32)):
        tolerance = largest * max(preserve.shape) * torch.finfo(dtype).eps
        rank = int((singular > tolerance).sum())
        null = vh[rank:].T.contiguous()
        null_dim = null.shape[1]
        notify('teacher_separation_'+precision)
        if null_dim:
            projected_teacher = teacher @ null
            projected_preserve = preserve @ null
            pu, ps, _ = svd(projected_teacher)
            # Absolute resolution tied to the preservation constraint. A tiny,
            # pure-roundoff null projection must not be promoted to useful rank
            # by taking a tolerance relative only to that tiny projection.
            resolved = int((ps > tolerance).sum())
            left = pu[:, :resolved]
            target_projection = left @ (left.T @ targets)
            teacher_fraction = float(projected_teacher.square().sum()/teacher_energy)
            preserve_fraction = float(projected_preserve.square().sum()/preserve_energy)
            artifact[precision+'_projected_teacher_singular_values'] = ps.cpu()
        else:
            resolved = 0
            target_projection = torch.zeros_like(targets)
            teacher_fraction = preserve_fraction = 0.0
            artifact[precision+'_projected_teacher_singular_values'] = torch.empty(0, dtype=torch.float64)
        residual = targets-target_projection
        per_teacher = []
        offset = 0
        for sid, count in zip(teacher_ids, rows_per_teacher):
            selected = slice(offset, offset+count)
            per_teacher.append(dict(state_id=sid,
                target_projection_residual_energy_ratio=float(residual[selected].square().sum()/targets[selected].square().sum().clamp_min(1e-300))))
            offset += count
        results[precision] = dict(rank=rank, null_dimensions=null_dim, absolute_singular_tolerance=tolerance,
            tolerance_formula='max(preservation_rows, columns) * source_precision_eps * largest_preservation_singular',
            projected_teacher_resolvable_rank=resolved,
            teacher_feature_energy_in_preservation_null_fraction=teacher_fraction,
            preservation_feature_energy_in_selected_null_fraction=preserve_fraction,
            teacher_target_projection_residual_energy_ratio=float(residual.square().sum()/target_energy),
            per_teacher_target_projection=per_teacher,
            meaning='FP64 null is numerical zero constraint; FP32 null permits precision-scale preservation leakage')
    summary = dict(preservation_rows=preserve.shape[0], feature_columns=preserve.shape[1],
        teacher_rows=teacher.shape[0], target_columns=targets.shape[1],
        qr_relative_residual=qr_residual, svd_relative_residual=svd_residual,
        largest_singular=largest, smallest_singular=float(singular[-1]),
        stable_rank=float(singular.square().sum()/singular[0].square().clamp_min(1e-300)),
        precision_conventions=results,
        scope='Fixed linear features and current TRAIN corpus; not a claim about nonlinear or trainable recurrent representations',
        new_readout_fit=False)
    return summary, artifact
