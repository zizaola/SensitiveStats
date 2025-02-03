# DataFrame methods and attributes
| Method                         | Description | Anlysed | Override? | Done? |
| -----------------------------  | ----------- | ------- | --------- | ----- |
|*Constructor*||||
|DataFrame([data, index, columns, dtype, copy]) |||||
|Two-dimensional, size-mutable, potentially |||||
|heterogeneous tabular data.|||||
| **Attributes and underlying data** | | | | |
| *Axes* | | |  | |
| DataFrame.index | The index (row labels) of the DataFrame.| | | |
| ataFrame.columns | The column labels of the DataFrame. | | | | 
| DataFrame.dtypes | Return the dtypes in the DataFrame. ||||
| DataFrame.info([verbose, buf, max_cols, ...]) | Print a concise summary of a DataFrame. ||||
| DataFrame.select_dtypes([include, exclude]) | Return a subset of the DataFrame's columns based on the column dtypes. ||||
| DataFrame.values | Return a Numpy representation of the DataFrame. ||||
DataFrame.axes | Return a list representing the axes of the DataFrame. |||||
| DataFrame.ndim | Return an int representing the number of axes / array dimensions. ||||
| DataFrame.size | Return an int representing the number of elements in this object. ||||
| DataFrame.shape | Return a tuple representing the dimensionality of the DataFrame. ||||
| DataFrame.memory_usage([index, deep]) | Return the memory usage of each column in bytes. ||||
| DataFrame.empty | Indicator whether Series/DataFrame is empty. ||||
| DataFrame.set_flags(*[, copy, ...]) | Return a new object with updated flags. ||||
|Conversion |||||
| DataFrame.astype(dtype[, copy, errors]) | Cast a pandas object to a specified dtype dtype. | ||||
| DataFrame.convert_dtypes([infer_objects, ...]) | Convert columns to the best possible dtypes using dtypes supporting pd.NA. ||||
| DataFrame.infer_objects([copy]) | Attempt to infer better dtypes for object columns. ||||
| DataFrame.copy([deep]) | Make a copy of this object's indices and data. | ||||
| DataFrame.bool() | (DEPRECATED) Return the bool of a single element Series or DataFrame. ||||
| DataFrame.to_numpy([dtype, copy, na_value]) | Convert the DataFrame to a NumPy array. ||||
| *Indexing, iteration* |||||
| DataFrame.head([n]) | Return the first n rows. ||||
| DataFrame.at | Access a single value for a row/column label pair. ||||
| DataFrame.iat | Access a single value for a row/column pair by integer position. ||||
| DataFrame.loc | Access a group of rows and columns by label(s) or a boolean array. ||||
| DataFrame.iloc | (DEPRECATED) Purely integer-location based indexing for selection by position. ||||
| DataFrame.insert(loc, column, value[, ...]) | Insert column into DataFrame at specified location. |||
| DataFrame.\_\_iter\_\_() |Iterate over info axis. ||||
| DataFrame.items() | Iterate over (column name, Series) pairs. ||||
| DataFrame.keys() | Get the 'info axis' (see Indexing for more). ||||
| DataFrame.iterrows() | Iterate over DataFrame rows as (index, Series) pairs. ||||
| DataFrame.itertuples([index, name]) | Iterate over DataFrame rows as namedtuples. ||||
| DataFrame.pop(item) | Return item and drop from frame. ||||
| DataFrame.tail([n]) | Return the last n rows. ||||
| DataFrame.xs(key[, axis, level, drop_level]) | Return cross-section from the Series/DataFrame. ||||
| DataFrame.get(key[, default]) | Get item from object for given key (ex: DataFrame column).||||
| DataFrame.isin(values) | Whether each element in the DataFrame is contained in values.||||
| DataFrame.where(cond[, other, inplace, ...]) | Replace values where the condition is False.||||
| DataFrame.mask(cond[, other, inplace, axis, ...]) | Replace values where the condition is True.||||
| DataFrame.query(expr, *[, inplace]) | Query the columns of a DataFrame with a boolean expression.||||
| For more information on .at, .iat, .loc, and .iloc, see the indexing documentation.||||
| *Binary operator functions* | ||||
| DataFrame.\_\_add\_\_(other) | Get Addition of DataFrame and other, column-wise. ||||
| DataFrame.add(other[, axis, level, fill_value]) | Get Addition of dataframe and other, element-wise (binary operator add). ||||
| DataFrame.sub(other[, axis, level, fill_value]) | Get Subtraction of dataframe and other, element-wise (binary operator sub). ||||
| DataFrame.mul(other[, axis, level, fill_value]) | Get Multiplication of dataframe and other, element-wise (binary operator mul). ||||
| DataFrame.div(other[, axis, level, fill_value]) | Get Floating division of dataframe and other, element-wise (binary operator truediv). ||||
| DataFrame.truediv(other[, axis, level, ...]) | Get Floating division of dataframe and other, element-wise (binary operator truediv). ||||
| DataFrame.floordiv(other[, axis, level, ...]) | Get Integer division of dataframe and other, element-wise (binary operator floordiv). ||||
| DataFrame.mod(other[, axis, level, fill_value]) | Get Modulo of dataframe and other, element-wise (binary operator mod). ||||
| DataFrame.pow(other[, axis, level, fill_value]) | Get Exponential power of dataframe and other, element-wise (binary operator pow). ||||
| DataFrame.dot(other) | Compute the matrix multiplication between the DataFrame and other. ||||
| DataFrame.radd(other[, axis, level, fill_value]) | Get Addition of dataframe and other, element-wise (binary operator radd). ||||
| DataFrame.rsub(other[, axis, level, fill_value]) | Get Subtraction of dataframe and other, element-wise (binary operator rsub). ||||
| DataFrame.rmul(other[, axis, level, fill_value]) | Get Multiplication of dataframe and other, element-wise (binary operator rmul). ||||
| DataFrame.rdiv(other[, axis, level, fill_value]) | Get Floating division of dataframe and other, element-wise (binary operator rtruediv). ||||
| DataFrame.rtruediv(other[, axis, level, ...]) | Get Floating division of dataframe and other, element-wise (binary operator rtruediv). ||||
| DataFrame.rfloordiv(other[, axis, level, ...]) | Get Integer division of dataframe and other, element-wise (binary operator rfloordiv). ||||
| DataFrame.rmod(other[, axis, level, fill_value]) | Get Modulo of dataframe and other, element-wise (binary operator rmod). ||||
| DataFrame.rpow(other[, axis, level, fill_value]) | Get Exponential power of dataframe and other, element-wise (binary operator rpow). ||||
| DataFrame.lt(other[, axis, level]) | Get Less than of dataframe and other, element-wise (binary operator lt). ||||
| DataFrame.gt(other[, axis, level]) | Get Greater than of dataframe and other, element-wise (binary operator gt). ||||
| DataFrame.le(other[, axis, level]) | Get Less than or equal to of dataframe and other, element-wise (binary operator le). ||||
| DataFrame.ge(other[, axis, level]) | Get Greater than or equal to of dataframe and other, element-wise (binary operator ge). ||||
| DataFrame.ne(other[, axis, level]) | Get Not equal to of dataframe and other, element-wise (binary operator ne). ||||
| DataFrame.eq(other[, axis, level]) | Get Equal to of dataframe and other, element-wise (binary operator eq). ||||
| DataFrame.combine(other, func[, fill_value, ...]) | Perform column-wise combine with another DataFrame. ||||
| DataFrame.combine_first(other) | Update null elements with value in the same location in other. ||||
| *Function application, GroupBy & window* ||||
| DataFrame.apply(func[, axis, raw, ...]) | Apply a function along an axis of the DataFrame. ||||
| DataFrame.map(func[, na_action]) | Apply a function to a Dataframe elementwise. ||||
| DataFrame.applymap(func[, na_action]) | (DEPRECATED) Apply a function to a Dataframe elementwise. ||||
| DataFrame.pipe(func, *args, **kwargs) | Apply chainable functions that expect Series or DataFrames.7 | DataFrame.agg([func, axis]) ||||
| Aggregate using one or more operations over the specifie | DataFrame.aggregate([func, axis]) ||||
| Aggregate using one or more operations over the specified axis. | DataFrame.transform(func[, axis]) ||||
| Call func on self producing a DataFrame with the same axis shape as self. | DataFrame.groupby([by, axis, level, ...]) ||||
| Group DataFrame using a mapper or by a Series of columns. | DataFrame.rolling(window[, min_periods, ...]) ||||
| Provide rolling window calculations. | DataFrame.expanding([min_periods, axis, method]) ||||
| Provide expanding window calculations. | DataFrame.ewm([com, span, halflife, alpha, ...]) ||||
| Provide exponentially weighted (EW) calculations.
| *Computations / descriptive stats* |||||
| DataFrame.abs() | Return a Series/DataFrame with absolute numeric value of each element. ||||
| DataFrame.all([axis, bool_only, skipna]) | Return whether all elements are True, potentially over an axis. ||||
| DataFrame.any(*[, axis, bool_only, skipna]) | Return whether any element is True, potentially over an axis. ||||
| DataFrame.clip([lower, upper, axis, inplace]) | Trim values at input threshold(s). ||||
| DataFrame.corr([method, min_periods, ...]) | Compute pairwise correlation of columns, excluding NA/null values. ||||
| DataFrame.corrwith(other[, axis, drop, ...]) | Compute pairwise correlation. ||||
| DataFrame.count([axis, numeric_only]) | Count non-NA cells for each column or row. ||||
| DataFrame.cov([min_periods, ddof, numeric_only]) | Compute pairwise covariance of columns, excluding NA/null values. ||||
| DataFrame.cummax([axis, skipna]) | Return cumulative maximum over a DataFrame or Series axis. ||||
| DataFrame.cummin([axis, skipna]) | Return cumulative minimum over a DataFrame or Series axis. ||||
| DataFrame.cumprod([axis, skipna]) | Return cumulative product over a DataFrame or Series axis. ||||
| DataFrame.cumsum([axis, skipna]) | Return cumulative sum over a DataFrame or Series axis. ||||
| DataFrame.describe([percentiles, include, ...]) | Generate descriptive statistics. ||||
| DataFrame.diff([periods, axis]) | First discrete difference of element. ||||
| DataFrame.eval(expr, *[, inplace]) | Evaluate a string describing operations on DataFrame columns. ||||
| DataFrame.kurt([axis, skipna, numeric_only]) | Return unbiased kurtosis over requested axis. ||||
| DataFrame.kurtosis([axis, skipna, numeric_only]) | Return unbiased kurtosis over requested axis. ||||
| DataFrame.max([axis, skipna, numeric_only]) | Return the maximum of the values over the requested axis. ||||
| DataFrame.mean([axis, skipna, numeric_only]) | Return the mean of the values over the requested axis. ||||
| DataFrame.median([axis, skipna, numeric_only]) | Return the median of the values over the requested axis. ||||
| DataFrame.min([axis, skipna, numeric_only]) | Return the minimum of the values over the requested axis. ||||
| DataFrame.mode([axis, numeric_only, dropna]) | Get the mode(s) of each element along the selected axis. ||||
| DataFrame.pct_change([periods, fill_method, ...]) | Fractional change between the current and a prior element. ||||
| DataFrame.prod([axis, skipna, numeric_only, ...]) | Return the product of the values over the requested axis. ||||
| DataFrame.product([axis, skipna, ...]) | Return the product of the values over the requested axis. ||||
| DataFrame.quantile([q, axis, numeric_only, ...]) | Return values at the given quantile over requested axis. ||||
| DataFrame.rank([axis, method, numeric_only, ...]) | Compute numerical data ranks (1 through n) along axis. ||||
| DataFrame.round([decimals]) | Round a DataFrame to a variable number of decimal places. ||||
| DataFrame.sem([axis, skipna, ddof, numeric_only]) | Return unbiased standard error of the mean over requested axis. ||||
| DataFrame.skew([axis, skipna, numeric_only]) | Return unbiased skew over requested axis. ||||
| DataFrame.sum([axis, skipna, numeric_only, ...]) | Return the sum of the values over the requested axis. ||||
| DataFrame.std([axis, skipna, ddof, numeric_only]) | Return sample standard deviation over requested axis. ||||
| DataFrame.var([axis, skipna, ddof, numeric_only]) | Return unbiased variance over requested axis. ||||
| DataFrame.nunique([axis, dropna]) | Count number of distinct elements in specified axis. ||||
| DataFrame.value_counts([subset, normalize, ...]) | Return a Series containing the frequency of each distinct row in the Dataframe. ||||
| *Reindexing / selection / label manipulation* |||||
| DataFrame.add_prefix(prefix[, axis]) | Prefix labels with string prefix. ||||
| DataFrame.add_suffix(suffix[, axis]) | Suffix labels with string suffix. ||||
| DataFrame.align(other[, join, axis, level, ...]) | Align two objects on their axes with the specified join method. ||||
| DataFrame.at_time(time[, asof, axis]) | Select values at particular time of day (e.g., 9:30AM). ||||
| DataFrame.between_time(start_time, end_time) | Select values between particular times of the day (e.g., 9:00-9:30 AM). ||||
| DataFrame.drop([labels, axis, index, ...]) | Drop specified labels from rows or columns. ||||
| DataFrame.drop_duplicates([subset, keep, ...]) | Return DataFrame with duplicate rows removed. ||||
| DataFrame.duplicated([subset, keep]) | Return boolean Series denoting duplicate rows. ||||
| DataFrame.equals(other) | Test whether two objects contain the same elements. ||||
| DataFrame.filter([items, like, regex, axis]) | Subset the dataframe rows or columns according to the specified index labels. ||||
| DataFrame.first(offset) | (DEPRECATED) Select initial periods of time series data based on a date offset. ||||
| DataFrame.head([n]) | Return the first n rows. ||||
| DataFrame.idxmax([axis, skipna, numeric_only]) | Return index of first occurrence of maximum over requested axis. ||||
| DataFrame.idxmin([axis, skipna, numeric_only]) | Return index of first occurrence of minimum over requested axis. ||||
| DataFrame.last(offset) | (DEPRECATED) Select final periods of time series data based on a date offset. ||||
| DataFrame.reindex([labels, index, columns, ...]) | Conform DataFrame to new index with optional filling logic. ||||
| DataFrame.reindex_like(other[, method, ...]) | Return an object with matching indices as other object. ||||
| DataFrame.rename([mapper, index, columns, ...]) | Rename columns or index labels. ||||
| DataFrame.rename_axis([mapper, index, ...]) | Set the name of the axis for the index or columns. ||||
| DataFrame.reset_index([level, drop, ...]) | Reset the index, or a level of it. ||||
| DataFrame.sample([n, frac, replace, ...]) | Return a random sample of items from an axis of object. ||||
| DataFrame.set_axis(labels, *[, axis, copy]) | Assign desired index to given axis. ||||
| DataFrame.set_index(keys, *[, drop, append, ...]) | Set the DataFrame index using existing columns. ||||
| DataFrame.tail([n]) | Return the last n rows. ||||
| DataFrame.take(indices[, axis]) | Return the elements in the given positional indices along an axis. ||||
| DataFrame.truncate([before, after, axis, copy]) | Truncate a Series or DataFrame before and after some index value. ||||
| *Missing data handling* |||||
| DataFrame.backfill(*[, axis, inplace, ...]) | (DEPRECATED) Fill NA/NaN values by using the next valid observation to fill the gap. ||||
| DataFrame.bfill(*[, axis, inplace, limit, ...]) | Fill NA/NaN values by using the next valid observation to fill the gap. ||||
| DataFrame.dropna(*[, axis, how, thresh, ...]) | Remove missing values. ||||
| DataFrame.ffill(*[, axis, inplace, limit, ...]) | Fill NA/NaN values by propagating the last valid observation to next valid. ||||
| DataFrame.fillna([value, method, axis, ...]) | Fill NA/NaN values using the specified method. ||||
| DataFrame.interpolate([method, axis, limit, ...]) | Fill NaN values using an interpolation method. ||||
| DataFrame.isna() | Detect missing values. ||||
| DataFrame.isnull() | DataFrame.isnull is an alias for DataFrame.isna. ||||
| DataFrame.notna() | Detect existing (non-missing) values. ||||
| DataFrame.notnull() | DataFrame.notnull is an alias for DataFrame.notna. ||||
| DataFrame.pad(*[, axis, inplace, limit, ...]) | (DEPRECATED) Fill NA/NaN values by propagating the last valid observation to next valid. ||||
| DataFrame.replace([to_replace, value, ...]) | Replace values given in to_replace with value. ||||
| *Reshaping, sorting, transposing*
| DataFrame.droplevel(level[, axis]) | Return Series/DataFrame with requested index / column level(s) removed. ||||
| DataFrame.pivot(*, columns[, index, values]) | Return reshaped DataFrame organized by given index / column values. ||||
| DataFrame.pivot_table([values, index, ...]) | Create a spreadsheet-style pivot table as a DataFrame. ||||
| DataFrame.reorder_levels(order[, axis]) | Rearrange index levels using input order. ||||
| DataFrame.sort_values(by, *[, axis, ...]) | Sort by the values along either axis. ||||
| DataFrame.sort_index(*[, axis, level, ...]) | Sort object by labels (along an axis). ||||
| DataFrame.nlargest(n, columns[, keep]) | Return the first n rows ordered by columns in descending order. ||||
| DataFrame.nsmallest(n, columns[, keep]) | Return the first n rows ordered by columns in ascending order. ||||
| DataFrame.swaplevel([i, j, axis]) | Swap levels i and j in a MultiIndex. ||||
| DataFrame.stack([level, dropna, sort, ...]) | Stack the prescribed level(s) from columns to index. ||||
| DataFrame.unstack([level, fill_value, sort]) | Pivot a level of the (necessarily hierarchical) index labels. ||||
| DataFrame.swapaxes(axis1, axis2[, copy]) | (DEPRECATED) Interchange axes and swap values axes appropriately. ||||
| DataFrame.melt([id_vars, value_vars, ...]) | Unpivot a DataFrame from wide to long format, optionally leaving identifiers set. ||||
| DataFrame.explode(column[, ignore_index]) | Transform each element of a list-like to a row, replicating index values. ||||
| DataFrame.squeeze([axis]) | Squeeze 1 dimensional axis objects into scalars. ||||
| DataFrame.to_xarray() | Return an xarray object from the pandas object. ||||
| DataFrame.T | The transpose of the DataFrame. ||||
| DataFrame.transpose(*args[, copy]) | Transpose index and columns. ||||
| *Combining / comparing / joining / merging* |||||
| DataFrame.assign(**kwargs) | Assign new columns to a DataFrame. ||||
| DataFrame.compare(other[, align_axis, ...]) | Compare to another DataFrame and show the differences. ||||
| DataFrame.join(other[, on, how, lsuffix, ...]) | Join columns of another DataFrame. ||||
| DataFrame.merge(right[, how, on, left_on, ...]) | Merge DataFrame or named Series objects with a database-style join. ||||
| DataFrame.update(other[, join, overwrite, ...]) | Modify in place using non-NA values from another DataFrame. ||||
| *Time Series-related* |||||
| DaFrame.asfreq(freq[, method, how, ...]) | Convert time series to specified frequency. ||||
| DataFrame.asof(where[, subset]) | Return the last row(s) without any NaNs before where. ||||
| DataFrame.shift([periods, freq, axis, ...]) | Shift index by desired number of periods with an optional time freq. ||||
| DataFrame.first_valid_index() | Return index for first non-NA value or None, if no non-NA value is found. ||||
| DataFrame.last_valid_index() | Return index for last non-NA value or None, if no non-NA value is found. ||||
| DataFrame.resample(rule[, axis, closed, ...]) | Resample time-series data. ||||
| DataFrame.to_period([freq, axis, copy]) | Convert DataFrame from DatetimeIndex to PeriodIndex. ||||
| DataFrame.to_timestamp([freq, how, axis, copy]) | Cast to DatetimeIndex of timestamps, at beginning of period. ||||
| DataFrame.tz_convert(tz[, axis, level, copy]) | Convert tz-aware axis to target time zone. ||||
| DataFrame.tz_localize(tz[, axis, level, ...]) | Localize tz-naive index of a Series or DataFrame to target time zone. ||||
| *Flags* |||||
| Flgs refer to attributes of the pandas object. |||||
| Properties of the dataset (like the date is was recorded, |||||
| the URL it was ac| essed from, etc.) should be stored in DataFrame.attrs. |||||
| Flags(obj, *, allows_duplicate_labels)  |||||
| Flags that apply to pandas objects.|||||
| *Metadata* |||||
| DataFrame.attrs is a dictionary for storing global metadata for this DataFrame. |||||
| Warning |||||
| DataFrame.attrs is considered experimental and may change without warning. |||||
| DataFrame.attrs |||||
| Dictionary of global attributes of this dataset.|||||
| *Plotting* |||||
| DataFrame.plot is both a callable method and a namespace |||||
| attribute for specific plotting methods of the form DataFrame.plot.<kind>. |||||
| DataFrame.plot([x, y, kind, ax, ....]) | DataFrame plotting accessor and method ||||
| DataFrame.plot.area([x, y, stacked]) | Draw a stacked area plot. ||||
| DataFrame.plot.bar([x, y]) | Vertical bar plot. ||||
| DataFrame.plot.barh([x, y]) | Make a horizontal bar plot. ||||
| DataFrame.plot.box([by]) | Make a box plot of the DataFrame columns. ||||
| DataFrame.plot.density([bw_method, ind]) | Generate Kernel Density Estimate plot using Gaussian kernels. ||||
| DataFrame.plot.hexbin(x, y[, C, ...]) | Generate a hexagonal binning plot. ||||
| DataFrame.plot.hist([by, bins]) | Draw one histogram of the DataFrame's columns. ||||
| DataFrame.plot.kde([bw_method, ind]) | Generate Kernel Density Estimate plot using Gaussian kernels. ||||
| DataFrame.plot.line([x, y]) | Plot Series or DataFrame as lines. ||||
| DataFrame.plot.pie(**kwargs) | Generate a pie plot. ||||
| DataFrame.plot.scatter(x, y[, s, c]) | Create a scatter plot with varying marker point size and color. ||||
| DataFrame.boxplot([column, by, ax, ...]) | Make a box plot from DataFrame columns. ||||
| DataFrame.hist([column, by, grid, ...]) | Make a histogram of the DataFrame's columns. ||||
| *Sparse accessor*
| parse-dtype specific methods and attributes are |||||
| provided under the DataFrame.sparse accessor. |||||
| DataFrame.sparse.density | Ratio of non-sparse points to total (dense) data points. ||||
| DataFrame.sparse.from_spmatrix(data[, ...]) | Create a new DataFrame from a scipy sparse matrix. ||||
| DataFrame.sparse.to_coo() | Return the contents of the frame as a sparse SciPy COO matrix. ||||
| DataFrame.sparse.to_dense() | Convert a DataFrame with sparse values to dense. ||||
 | *Serialization / IO / conversion* ||||| 
| DataFrame.from_dict(data[, orient, dtype, ...]) | Construct DataFrame from dict of array-like or dicts. ||||
| DataFrame.from_records(data[, index, ...]) | Convert structured or record ndarray to DataFrame. ||||
| DataFrame.to_orc([path, engine, index, ...]) | Write a DataFrame to the ORC format. ||||
| DataFrame.to_parquet([path, engine, ...]) | Write a DataFrame to the binary parquet format. ||||
| DataFrame.to_pickle(path, *[, compression, ...]) | Pickle (serialize) object to file. ||||
| DataFrame.to_csv([path_or_buf, sep, na_rep, ...]) | Write object to a comma-separated values (csv) file. ||||
| DataFrame.to_hdf(path_or_buf, *, key[, ...]) | Write the contained data to an HDF5 file using HDFStore. ||||
| DataFrame.to_sql(name, con, *[, schema, ...]) | Write records stored in a DataFrame to a SQL database. ||||
| DataFrame.to_dict([orient, into, index]) | Convert the DataFrame to a dictionary. ||||
| DataFrame.to_excel(excel_writer, *[, ...]) | Write object to an Excel sheet. ||||
| DataFrame.to_json([path_or_buf, orient, ...]) | Convert the object to a JSON string. ||||
| DataFrame.to_html([buf, columns, col_space, ...]) | Render a DataFrame as an HTML table. ||||
| DataFrame.to_feather(path, **kwargs) | Write a DataFrame to the binary Feather format. ||||
| DataFrame.to_latex([buf, columns, header, ...]) | Render object to a LaTeX tabular, longtable, or nested table. ||||
| DataFrame.to_stata(path, *[, convert_dates, ...]) | Export DataFrame object to Stata dta format. ||||
| DataFrame.to_gbq(destination_table, *[, ...]) | (DEPRECATED) Write a DataFrame to a Google BigQuery table. ||||
| DataFrame.to_records([index, column_dtypes, ...]) | Convert DataFrame to a NumPy record array. ||||
| DataFrame.to_string([buf, columns, ...]) | Render a DataFrame to a console-friendly tabular output. ||||
| DataFrame.to_clipboard(*[, excel, sep]) | Copy object to the system clipboard. ||||
| DataFrame.to_markdown([buf, mode, index, ...]) | Print DataFrame in Markdown-friendly format. ||||
| DataFrame.style | Returns a Styler object. ||||
| DataFrame.\_\_dataframe\_\_([nan_as_null, ...]) | Return the dataframe interchange object implementing the interchange protocol. ||||
| *magic methods* |||||
| T | str(object='') -> str ||||
| _AXIS_LEN | str(object='') -> str ||||
| _AXIS_ORDERS | str(object='') -> str ||||
| _AXIS_TO_AXIS_NUMBER | str(object='') -> str ||||
| _HANDLED_TYPES | str(object='') -> str ||||
| \_\_abs\_\_ | str(object='') -> str ||||
| \_\_add\_\_ | str(object='') -> str ||||
| \_\_and\_\_ | str(object='') -> str ||||
| \_\_annotations\_\_ | str(object='') -> str ||||
| \_\_array\_\_ | str(object='') -> str ||||
| \_\_array_priority\_\_ | str(object='') -> str ||||
| \_\_array_ufunc\_\_ | str(object='') -> str ||||
| \_\_arrow_c_stream\_\_ | str(object='') -> str ||||
| \_\_bool\_\_ | str(object='') -> str ||||
| \_\_class\_\_ | str(object='') -> str ||||
| \_\_contains\_\_ | str(object='') -> str ||||
| \_\_copy\_\_ | str(object='') -> str ||||
| \_\_dataframe\_\_ | str(object='') -> str ||||
| \_\_dataframe_consortium_standard\_\_ | str(object='') -> str ||||
| \_\_deepcopy\_\_ | str(object='') -> str ||||
| \_\_delattr\_\_ | str(object='') -> str ||||
| \_\_delitem\_\_ | str(object='') -> str ||||
| \_\_dict\_\_ | str(object='') -> str ||||
| \_\_dir\_\_ | str(object='') -> str ||||
| \_\_divmod\_\_ | str(object='') -> str ||||
| \_\_doc\_\_ | str(object='') -> str ||||
| \_\_eq\_\_ | str(object='') -> str ||||
| \_\_finalize\_\_ | str(object='') -> str ||||
| \_\_floordiv\_\_ | str(object='') -> str ||||
| \_\_format\_\_ | str(object='') -> str ||||
| \_\_ge\_\_ | str(object='') -> str ||||
| \_\_getattr\_\_ | str(object='') -> str ||||
| \_\_getattribute\_\_ | str(object='') -> str ||||
| \_\_getitem\_\_ | str(object='') -> str ||||
| \_\_getstate\_\_ | str(object='') -> str ||||
| \_\_gt\_\_ | str(object='') -> str ||||
| \_\_hash\_\_ | str(object='') -> str ||||
| \_\_iadd\_\_ | str(object='') -> str ||||
| \_\_iand\_\_ | str(object='') -> str ||||
| \_\_ifloordiv\_\_ | str(object='') -> str ||||
| \_\_imod\_\_ | str(object='') -> str ||||
| \_\_imul\_\_ | str(object='') -> str ||||
| \_\_init\_\_ | str(object='') -> str ||||
| \_\_init_subclass\_\_ | str(object='') -> str ||||
| \_\_invert\_\_ | str(object='') -> str ||||
| \_\_ior\_\_ | str(object='') -> str ||||
| \_\_ipow\_\_ | str(object='') -> str ||||
| \_\_isub\_\_ | str(object='') -> str ||||
| \_\_iter\_\_ | str(object='') -> str ||||
| \_\_itruediv\_\_ | str(object='') -> str ||||
| \_\_ixor\_\_ | str(object='') -> str ||||
| \_\_le\_\_ | str(object='') -> str ||||
| \_\_len\_\_ | str(object='') -> str ||||
| \_\_lt\_\_ | str(object='') -> str ||||
| \_\_matmul\_\_ | str(object='') -> str ||||
| \_\_mod\_\_ | str(object='') -> str ||||
| \_\_module\_\_ | str(object='') -> str ||||
| \_\_mul\_\_ | str(object='') -> str ||||
| \_\_ne\_\_ | str(object='') -> str ||||
| \_\_neg\_\_ | str(object='') -> str ||||
| \_\_new\_\_ | str(object='') -> str ||||
| \_\_nonzero\_\_ | str(object='') -> str ||||
| \_\_or\_\_ | str(object='') -> str ||||
| \_\_pandas_priority\_\_ | str(object='') -> str ||||
| \_\_pos\_\_ | str(object='') -> str ||||
| \_\_pow\_\_ | str(object='') -> str ||||
| \_\_radd\_\_ | str(object='') -> str ||||
| \_\_rand\_\_ | str(object='') -> str ||||
| \_\_rdivmod\_\_ | str(object='') -> str ||||
| \_\_reduce\_\_ | str(object='') -> str ||||
| \_\_reduce_ex\_\_ | str(object='') -> str ||||
| \_\_repr\_\_ | str(object='') -> str ||||
| \_\_rfloordiv\_\_ | str(object='') -> str ||||
| \_\_rmatmul\_\_ | str(object='') -> str ||||
| \_\_rmod\_\_ | str(object='') -> str ||||
| \_\_rmul\_\_ | str(object='') -> str ||||
| \_\_ror\_\_ | str(object='') -> str ||||
| \_\_round\_\_ | str(object='') -> str ||||
| \_\_rpow\_\_ | str(object='') -> str ||||
| \_\_rsub\_\_ | str(object='') -> str ||||
| \_\_rtruediv\_\_ | str(object='') -> str ||||
| \_\_rxor\_\_ | str(object='') -> str ||||
| \_\_setattr\_\_ | str(object='') -> str ||||
| \_\_setitem\_\_ | str(object='') -> str ||||
| \_\_setstate\_\_ | str(object='') -> str ||||
| \_\_sizeof\_\_ | str(object='') -> str ||||
| \_\_str\_\_ | str(object='') -> str ||||
| \_\_sub\_\_ | str(object='') -> str ||||
| \_\_subclasshook\_\_ | str(object='') -> str ||||
| \_\_truediv\_\_ | str(object='') -> str ||||
| \_\_weakref\_\_ | str(object='') -> str ||||
| \_\_xor\_\_ | str(object='') -> str ||||
| _accessors | str(object='') -> str ||||
| _accum_func | str(object='') -> str ||||
| _agg_examples_doc | str(object='') -> str ||||
| _agg_see_also_doc | str(object='') -> str ||||
| _align_for_op | str(object='') -> str ||||
| _align_frame | str(object='') -> str ||||
| _align_series | str(object='') -> str ||||
| _append | str(object='') -> str ||||
| _arith_method | str(object='') -> str ||||
| _arith_method_with_reindex | str(object='') -> str ||||
| _as_manager | str(object='') -> str ||||
| _attrs | str(object='') -> str ||||
| _box_col_values | str(object='') -> str ||||
| _can_fast_transpose | str(object='') -> str ||||
| _check_inplace_and_allows_duplicate_labels | str(object='') -> str ||||
| _check_is_chained_assignment_possible | str(object='') -> str ||||
| _check_label_or_level_ambiguity | str(object='') -> str ||||
| _check_setitem_copy | str(object='') -> str ||||
| _clear_item_cache | str(object='') -> str ||||
| _clip_with_one_bound | str(object='') -> str ||||
| _clip_with_scalar | str(object='') -> str ||||
| _cmp_method | str(object='') -> str ||||
| _combine_frame | str(object='') -> str ||||
| _consolidate | str(object='') -> str ||||
| _consolidate_inplace | str(object='') -> str ||||
| _construct_axes_dict | str(object='') -> str ||||
| _construct_result | str(object='') -> str ||||
| _constructor | str(object='') -> str ||||
| _constructor_from_mgr | str(object='') -> str ||||
| _constructor_sliced | str(object='') -> str ||||
| _constructor_sliced_from_mgr | str(object='') -> str ||||
| _create_data_for_split_and_tight_to_dict | str(object='') -> str ||||
| _data | str(object='') -> str ||||
| _deprecate_downcast | str(object='') -> str ||||
| _dir_additions | str(object='') -> str ||||
| _dir_deletions | str(object='') -> str ||||
| _dispatch_frame_op | str(object='') -> str ||||
| _drop_axis | str(object='') -> str ||||
| _drop_labels_or_levels | str(object='') -> str ||||
| _ensure_valid_index | str(object='') -> str ||||
| _find_valid_index | str(object='') -> str ||||
| _flags | str(object='') -> str ||||
| _flex_arith_method | str(object='') -> str ||||
| _flex_cmp_method | str(object='') -> str ||||
| _from_arrays | str(object='') -> str ||||
| _from_mgr | str(object='') -> str ||||
| _get_agg_axis | str(object='') -> str ||||
| _get_axis | str(object='') -> str ||||
| _get_axis_name | str(object='') -> str ||||
| _get_axis_number | str(object='') -> str ||||
| _get_axis_resolvers | str(object='') -> str ||||
| _get_block_manager_axis | str(object='') -> str ||||
| _get_bool_data | str(object='') -> str ||||
| _get_cleaned_column_resolvers | str(object='') -> str ||||
| _get_column_array | str(object='') -> str ||||
| _get_index_resolvers | str(object='') -> str ||||
| _get_item_cache | str(object='') -> str ||||
| _get_label_or_level_values | str(object='') -> str ||||
| _get_numeric_data | str(object='') -> str ||||
| _get_value | str(object='') -> str ||||
| _get_values_for_csv | str(object='') -> str ||||
| _getitem_bool_array | str(object='') -> str ||||
| _getitem_multilevel | str(object='') -> str ||||
| _getitem_nocopy | str(object='') -> str ||||
| _getitem_slice | str(object='') -> str ||||
| _gotitem | str(object='') -> str ||||
| _hidden_attrs | str(object='') -> str ||||
| _indexed_same | str(object='') -> str ||||
| _info_axis | str(object='') -> str ||||
| _info_axis_name | str(object='') -> str ||||
| _info_axis_number | str(object='') -> str ||||
| _info_repr | str(object='') -> str ||||
| _init_mgr | str(object='') -> str ||||
| _inplace_method | str(object='') -> str ||||
| _internal_names | str(object='') -> str ||||
| _internal_names_set | str(object='') -> str ||||
| _is_copy | str(object='') -> str ||||
| _is_homogeneous_type | str(object='') -> str ||||
| _is_label_or_level_reference | str(object='') -> str ||||
| _is_label_reference | str(object='') -> str ||||
| _is_level_reference | str(object='') -> str ||||
| _is_mixed_type | str(object='') -> str ||||
| _is_view | str(object='') -> str ||||
| _is_view_after_cow_rules | str(object='') -> str ||||
| _iset_item | str(object='') -> str ||||
| _iset_item_mgr | str(object='') -> str ||||
| _iset_not_inplace | str(object='') -> str ||||
| _item_cache | str(object='') -> str ||||
| _iter_column_arrays | str(object='') -> str ||||
| _ixs | str(object='') -> str ||||
| _logical_func | str(object='') -> str ||||
| _logical_method | str(object='') -> str ||||
| _maybe_align_series_as_frame | str(object='') -> str ||||
| _maybe_cache_changed | str(object='') -> str ||||
| _maybe_update_cacher | str(object='') -> str ||||
| _metadata | str(object='') -> str ||||
| _mgr | str(object='') -> str ||||
| _min_count_stat_function | str(object='') -> str ||||
| _needs_reindex_multi | str(object='') -> str ||||
| _pad_or_backfill | str(object='') -> str ||||
| _protect_consolidate | str(object='') -> str ||||
| _reduce | str(object='') -> str ||||
| _reduce_axis1 | str(object='') -> str ||||
| _reindex_axes | str(object='') -> str ||||
| _reindex_multi | str(object='') -> str ||||
| _reindex_with_indexers | str(object='') -> str ||||
| _rename | str(object='') -> str ||||
| _replace_columnwise | str(object='') -> str ||||
| _repr_data_resource_ | str(object='') -> str ||||
| _repr_fits_horizontal_ | str(object='') -> str ||||
| _repr_fits_vertical_ | str(object='') -> str ||||
| _repr_html_ | str(object='') -> str ||||
| _repr_latex_ | str(object='') -> str ||||
| _reset_cache | str(object='') -> str ||||
| _reset_cacher | str(object='') -> str ||||
| _sanitize_column | str(object='') -> str ||||
| _series | str(object='') -> str ||||
| _set_axis | str(object='') -> str ||||
| _set_axis_name | str(object='') -> str ||||
| _set_axis_nocheck | str(object='') -> str ||||
| _set_is_copy | str(object='') -> str ||||
| _set_item | str(object='') -> str ||||
| _set_item_frame_value | str(object='') -> str ||||
| _set_item_mgr | str(object='') -> str ||||
| _set_value | str(object='') -> str ||||
| _setitem_array | str(object='') -> str ||||
| _setitem_frame | str(object='') -> str ||||
| _setitem_slice | str(object='') -> str ||||
| _shift_with_freq | str(object='') -> str ||||
| _should_reindex_frame_op | str(object='') -> str ||||
| _slice | str(object='') -> str ||||
| _stat_function | str(object='') -> str ||||
| _stat_function_ddof | str(object='') -> str ||||
| _take_with_is_copy | str(object='') -> str ||||
| _to_dict_of_blocks | str(object='') -> str ||||
| _to_latex_via_styler | str(object='') -> str ||||
| _typ | str(object='') -> str ||||
| _update_inplace | str(object='') -> str ||||
| _validate_dtype | str(object='') -> str ||||
| _values | str(object='') -> str ||||
| _where | str(object='') -> str ||||