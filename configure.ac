AC_INIT([WideLM], 0.1-1)

AC_LANG(C++)
AC_REQUIRE_CPP


AC_MSG_NOTICE([checking for critical tools used by configure])

AC_ARG_WITH([compiler-bindir],
	AC_HELP_STRING([--with-compiler-bindir=COMPILER_BINDIR],
		[specify the directory containing gcc, e.g. /usr/bin]),
	[COMPILER_BINDIR="$withval"])

AC_ARG_WITH([cuda-home],
	AC_HELP_STRING([--with-cuda-home=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]),
	[CUDA_HOME="$withval"])


AC_MSG_CHECKING("environment variable COMPILER_BINDIR")
if test -z "${COMPILER_BINDIR}"; then
	AC_MSG_RESULT("COMPILER_BINDIR not set; using user's default path")
else
	AC_MSG_RESULT("using COMPILER_BINDIR=${COMPILER_BINDIR}")
	AC_CHECK_FILE([${COMPILER_BINDIR}/gcc], [HAS_GCC="yes"])
	if test -z "${HAS_GCC}"; then
		AC_MSG_ERROR("cannot find gcc; check that ${COMPILER_BINDIR} is the correct location; using user's default path")
		CU_PARAMS=""
	else
		AC_MSG_RESULT("gcc found in ${COMPILER_BINDIR}")
		CU_PARAMS="--compiler-bindir ${COMPILER_BINDIR}"
	fi
fi

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
   	CUDA_HOME="/usr/local/cuda"
	AC_MSG_RESULT("CUDA_HOME not set; using default ${CUDA_HOME}")
else
	AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi


AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
	AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
	AC_MSG_RESULT("nvcc found")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
	AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
	CU_LIBDIR="${CUDA_HOME}/lib"
else
	AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
	CU_LIBDIR="${CUDA_HOME}/lib64"
fi

# Much of what follows has been pilferred from RQuantLib package
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([Could not determine R_HOME.])   
fi

R_INCL=`"${R_HOME}/bin/R" CMD config --cppflags`
R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
R_CPIC=`"${R_HOME}/bin/R" CMD config CPICFLAGS`

## look for Rscript, but use the one found via R_HOME to allow for multiple installations
AC_DEFUN(AC_PROG_RSCRIPT, [AC_CHECK_PROG(RSCRIPT,Rscript,yes)])
AC_PROG_RSCRIPT

if test x"${RSCRIPT}" == x"yes" ; then
     rcpp_ldflags=`${R_HOME}/bin/Rscript -e 'Rcpp:::LdFlags()'`
     AC_SUBST([RCPP_LDFLAGS],["${LDFLAGS} $rcpp_ldflags"])
     rcpp_incl=`${R_HOME}/bin/Rscript -e 'Rcpp:::CxxFlags()'`
     AC_SUBST([RCPP_INCL],["${LDFLAGS} $rcpp_incl"])
else
    echo "
  Your installation does not appear to have Rscript installed.

  Please make sure that you have a working and complete R installation.
"
    exit 1
fi

AC_MSG_CHECKING("building the nvcc command line")
NVCC="${CUDA_HOME}/bin/nvcc"

AC_MSG_NOTICE([determining host platform])
arch=`uname -m`
osrelease=`uname -r`
osversion=`uname -v`

AC_MSG_NOTICE([Building Makefile])
AC_SUBST(CU_LIBDIR)
AC_SUBST(CU_PARAMS)
AC_SUBST(CUDA_HOME)
AC_SUBST(NVCC)
AC_SUBST(R_INCL)
AC_SUBST(R_LIB)
AC_SUBST(R_CPIC)
AC_SUBST(RCPP_LDFLAGS)
AC_SUBST(RCPP_INCL)
AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
