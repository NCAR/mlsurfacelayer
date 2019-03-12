module module_sf_sfclay_random_forest
    contains

    subroutine sfclay_random_forest(u3d, v3d, t3d, qv3d, p3d, dz8w, mavail, &
                                    xland, tsk, psfc, &
                                    ust, zol, mol, rmol, hfx, qfx, lh, &
                                    u10, v10, th2, t2, q2, qsfc, &
                                    ids, ide, jds, jde, kds, kde,                    &
                                    ims, ime, jms, jme, kms, kme,                    &
                                    its, ite, jts, jte, kts, kte)
        implicit none
        integer, intent(in) :: ids, ide, jds, jde, kds, kde, &
                ims, ime, jms, jme, kms, kme, &
                its, ite, jts, jte, kts, kte
        real, dimension(ims:ime, kms:kme, jms:jme), &
                intent(in) :: dz8w

        real, dimension(ims:ime, kms:kme, jms:jme), &
                intent(in) :: qv3d, &
                p3d, &
                t3d

        real, dimension(ims:ime, jms:jme), &
                intent(in) :: mavail, &
                xland, &
                tsk
        real, dimension(ims:ime, jms:jme), &
                intent(out) :: u10, &
                v10, &
                th2, &
                t2, &
                q2, &
                qsfc

        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: regime, &
                hfx, &
                qfx, &
                lh, &
                mol, rmol

        real, dimension(its:ite) :: u1d, &
                v1d, &
                qv1d, &
                p1d, &
                t1d

        real, dimension(its:ite) :: dz8w1d

        integer :: i, j

        do j = jts, jte
            do i = its, ite
                dz8w1d(i) = dz8w(i, 1, j)
            enddo

            do i = its, ite
                u1d(i) = u3d(i, 1, j)
                v1d(i) = v3d(i, 1, j)
                qv1d(i) = qv3d(i, 1, j)
                p1d(i) = p3d(i, 1, j)
                t1d(i) = t3d(i, 1, j)
            enddo

        enddo
    end subroutine sfclay_random_forest

    subroutine sfclay_random_forest_1d()

    end subroutine sfclay_random_forest_1d
end module module_sf_sfclay_random_forest