regrank.solvers.firstOrderMethods
=================================

.. py:module:: regrank.solvers.firstOrderMethods

.. autoapi-nested-parse::

   firstOrderMethods module
       Mostly for APPM 5630 at CU Boulder, but others may find it useful too

       The main routine is gradientDescent(...) which can also do proximal/projected
           gradient descent as well as Nesterov acceleration (i.e., it can do FISTA)

       Also includes lassoSolver(...) which is a wrapper around gradientDescent(...)
           to solve the lasso problem min_x .5||Ax-b||^2 + tau ||x||_1

       Other routines:

           createTestProblem(...) and runAllTestProblems() have some test problems,
               including least-squares, lasso, and logistic regression

           backtrackingLinesearch(...) and LipschitzLinesearch(...) as well as
           powerMethod(...) are utilities

       Note: not very well documented, but hopefully simple enough that you can figure
           things out

       The test problems rely on cvxpy, and the logistic problem relies on scipy.special
       The main module depends heavily on numpy

       Finally, if you run this file from the command line, it will execute the tests

       Features to add:
           (1) adaptive restarts. Done, 4/26/23
           (2) take advantage of functions that give you function value and gradient
               at the same time (since it's often possible to share some computation;
               e.g., f(x) = 1/2||Ax-b||^2, grad(x) = A'*(Ax-b), the residual Ax-b
               only needs to be computed once. [Along with this, make a class for
               fcn/grad computation that records total # of fcn calls]

       Stephen Becker, April 1 2021, stephen.becker@colorado.edu
       Updates (Douglas-Rachford, bookkeeper) April 2023
       TODO:
         incorporate the bookkeeper class into the gradient descent code
           (this makes the actual algorithm itself more clear)
         in gradient descent code, is the function value properly updated? May affect linesearches

       Released under the Modified BSD License:

   Copyright (c) 2023, Stephen Becker. All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
   3. Neither the name of the Stephen Becker nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL STEPHEN BECKER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE



Classes
-------

.. autoapisummary::

   regrank.solvers.firstOrderMethods.bookkeeper


Functions
---------

.. autoapisummary::

   regrank.solvers.firstOrderMethods.as_column_vec
   regrank.solvers.firstOrderMethods.print_status
   regrank.solvers.firstOrderMethods.backtrackingLinesearch
   regrank.solvers.firstOrderMethods.LipschitzLinesearch
   regrank.solvers.firstOrderMethods.LipschitzLinesearch_stabler
   regrank.solvers.firstOrderMethods.powerMethod
   regrank.solvers.firstOrderMethods.DouglasRachford
   regrank.solvers.firstOrderMethods.gradientDescent


Module Contents
---------------

.. py:function:: as_column_vec(x)

   Input x is of size (n,) or (n,1) or (1,n)
   and output is always of size (n,1). This lets us be consistent


.. py:function:: print_status(prob, x)

.. py:function:: backtrackingLinesearch(f, x, p, grad, t, fx=None, c=1e-06, rho=0.9, linesearchMaxIters=None)

   "
   Backtracking linesearch, testing with the Armijo Condition
   f    is function to evaluate objective function
   x    is current point
   p    is search direction (often the negative gradient)
   grad is the gradient
   t    is the initial guess for a stepsize
   fx   is f(x) (for the value of x passed in) [optional]
   :returns: x,t,fx,iter   where x is new point, t is the stepsize used, fx=f(x)


.. py:function:: LipschitzLinesearch(f, x, grad, t, fx=None, prox=None, rho=0.9, linesearchMaxIters=None)

   "
   Backtracking linesearch, should work if f is Lipschitz
     Note: if we are minimizing f + g via proximal gradient methods,
     then f should be just f, not f+g
   f    is function to evaluate objective function
   x    is current point
   grad is the gradient
   t    is the initial guess for a stepsize
   fx   is f(x) (for the value of x passed in) [optional]
   :returns: x,t,fx,iter   where x is new point, t is the stepsize used, fx=f(x)


.. py:function:: LipschitzLinesearch_stabler(f, x, g, t, fx=None, gx=None, prox=None, rho=0.9, linesearchMaxIters=None)

   "
   Backtracking linesearch, should work if f is Lipschitz
     Note: if we are minimizing f + g via proximal gradient methods,
     then f should be just f, not f+g
   f    is function to evaluate objective function
   x    is current point
   g is the gradient (a function)
   t    is the initial guess for a stepsize
   fx   is f(x) (for the value of x passed in) [optional]
   gx   is grad(x) (for the value of x passed in) [optional]
   :returns: x,t,fx,iter   where x is new point, t is the stepsize used, fx=f(x)

   More stable version (for numerical rounding errors)
     but requires an additional gradient evaluation
   This is Eq (5.7) in https://amath.colorado.edu/faculty/becker/TFOCS.pdf
     (whereas the other LipschitzLinesearch is eq (5.6) )
     "Templates for Convex Cone Problems with Applications to Sparse Signal Recovery"
     by S. Becker, E. Candès, M. Grant. Mathematical Programming Computation, 3(3) 2011, pp 165–21


.. py:function:: powerMethod(A, At=None, domainSize=None, x=None, iters=100, tol=1e-06, rng=None, quiet=False)

.. py:class:: bookkeeper(printEvery, errorFunction, F, printStepsize=True, tol=1e-06, tolAbs=None, tolX=None, tolG=None, tolErr=-1, minIter=1)

   .. py:attribute:: printEvery


   .. py:attribute:: errFcn


   .. py:attribute:: objFcn


   .. py:attribute:: errHist
      :value: []



   .. py:attribute:: fcnHist
      :value: []



   .. py:attribute:: printStepsize
      :value: True



   .. py:attribute:: stoppingFlag
      :value: 'Reached max iterations'



   .. py:attribute:: tol
      :value: 1e-06



   .. py:attribute:: tolX
      :value: None



   .. py:attribute:: tolG
      :value: None



   .. py:attribute:: tolAbs
      :value: None



   .. py:attribute:: tolErr
      :value: -1



   .. py:attribute:: minIter
      :value: 1



   .. py:method:: printInitialization()


   .. py:method:: update_and_print(x, k, stepsize=None, ignorePrintEvery=False, Fx=None)

      x is current iterate, k is stepnumber, stepsize is stepsize



   .. py:method:: finalize(x, k, stepsize=None)


   .. py:method:: checkStoppingCondition(x, xOld=None, iteration=np.inf, gradient=None, stepsize=None)


.. py:function:: DouglasRachford(prox1, prox2, y0, gamma=1, F=None, overrelax=1, tol=1e-06, maxIters=500, printEvery=10, errorFunction=None)

   Douglas Rachford algorithm to minimize F(x) = f1(x) + f2(x)


.. py:function:: gradientDescent(f, grad, x0, prox=None, prox_obj=None, stepsize=None, tol=1e-06, maxIters=10000.0, printEvery=None, linesearch=False, stepsizeOptimism=1.1, errorFunction=None, ArmijoLinesearch=None, LipschitzStable=True, saveHistory=False, acceleration=True, restart=-5, **kwargs)

   (Proximal) gradient descent with either fixed stepsize or backtracking linesearch
   Minimizes F(x) := f(x) + g(x), where f is differentiable and f has an easy proximity operator
     (if g=0 then this reduces to gradient descent)

   f                 is smooth part of the objective function
   grad              returns gradient of f
   x0                is initial starting point
   prox              proximity operator for a function g,  prox(y,t) = argmin_x g(x) + 1/(2*t)||x-y||^2
   prox_obj          aka g(x), this is when we solve min_x f(x) + g(x)
   stepsize          either a scalar or if None (default) then uses backtracking linesearch
   linesearch        if True then uses backtracking linesearch (default: true if stepsize is None)
   ArmijoLinesearch  if True, uses Armijo backgracking linesearch (default: true, if no prox and no acceleration, otherwise false)
   LipschitzStable   if not using Armijo linesearch, then use the stable (slightly more expensive) linesearch
   tol               stopping tolerance
   maxIters          maximum number of iterations
   printEvery        prints out information every printEvery steps; set to 0 for quiet
   stepsizeOptimism  how much to multiply old stepsize by when guessing new stepsize (linesearch only)
   errorFunction     if provided, will evaluate errorFunction(x) at every iteration
   saveHistory       whether to save function and error history
   acceleration      Nesterov acceleration (default: True)
   restart           How often to restart acceleration; if negative, then adaptive restart

   Outputs:
   x         final iterate
   data      dictionary with detailed info. Keys include:
     'steps', 'fcnHistory', 'errHistory', 'flag', 'fx'

     Stephen Becker, University of Colorado Boulder, March 2021 and April 2023
