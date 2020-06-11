FAQs
====

What if my angle of incidence is 0?
-----------------------------------

If your angle of incidence is 0 degrees, then you probably only have two data files ("parallel" and "perpendicular"), rather than the typical four data files ("PP", "PS", "SP", and "SS") which are used in the examples and tutorials. There are two ways to go about this.

From the formula side
.....................

This is the easier method. After having loaded your SHG data like normal (i.e. not coercing your data to oblique incidence):

>>> data_filenames_dict = {
>>>     'PP':'dataPP.csv',
>>>     'SP':'dataSP.csv',
>>> }
>>> dat, fdat = shgpy.load_data_and_fourier_transform(data_filenames_dict, 'degrees', normal_to_oblique=False)

only use the selected components of the Fourier formula to fit the data:

>>> full_fform = shgpy.load_fform('my_filename.p')
>>> iterable = {}
>>> for k,v in full_fform.get_items():
>>>     if k in ['PP', 'SP']:
>>>         iterable[k] = v
>>> partial_fform = shgpy.fFormContainer(iterable)

now just use ``partial_fform`` to fit the data like you would normally, as in the :doc:`fitting tutorial <tutorial/fitting_tutorial>`.

From the data side
..................

From the data side, you can always artificially rotate your parallel and perpendicular datasets by 90 degrees, and call those the missing polarization combinations. This is achieved by adding the ``normal_to_oblique=True`` flag in whatever function you are using to load your data. Ultimately, this causes the ``__init__`` method of :class:`shgpy.core.data_handler.DataContainer` to rotate your data appropriately and spit out a `DataContainer` with four polarization combinations -- ``'PP'``, ``'PS'``, ``'SP'``, and ``'SS'``. From there you can use the rest of ``shgpy`` as normal.

Note that in order for this mapping to be well-defined, you need to specify exactly *which* polarizer you rotated to go from parallel to perpendicular. This is indicated by the keys in your ``data_filenames_dict``; for example, if you input

>>> data_filenames_dict = {
>>>     'PP':'dataPP.csv',
>>>     'SP':'dataSP.csv',
>>> }

this is telling ``shgpy`` that you rotated your incoming polarizer, whereas if you had input 

>>> data_filenames_dict = {
>>>     'PP':'dataPP.csv',
>>>     'PS':'dataPS.csv',
>>> }

this indicates that the outgoing polarizer was rotated.

Remember of course to set ``aoi=0`` when you generate the unconctrated Fourier transforms (see the :doc:`fitting tutorial <tutorial/fitting_tutorial>`).

How do I view my fitting formula?
---------------------------------

Once you've generated your fitting formula using :func:`shgpy.fformgen.generate_contracted_fourier_transforms`, sometimes you want to literally visualize the formula that you're fitting to. In that case, you can use :func:`shgpy.core.data_handler.fFormContainer.get_pc` to view the formula in Fourier space, or use 

>>> form = shgpy.fform_to_form(fform)

to convert it to a ``phi``-space formula and then :func:`shgpy.core.data_handler.FormContainer.get_pc` to view it.

See the example file at ``examples/look_at_fformula_example.py``.
