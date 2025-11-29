# coding=utf-8
# Copyright 2024 HuggingFace Inc. team and ACE contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ace_tools_open integration for smolagents.

Provides GPT-4o style tools for data display and analysis,
compatible with the ace_tools_open package.

Based on: https://github.com/zinccat/ace_tools_open
"""

from typing import Any, Optional, List
from ..tools import tool


@tool
def display_dataframe_to_user(df: Any, title: str = "Data") -> str:
    """
    Display a pandas DataFrame to the user in a formatted way.

    This tool renders a DataFrame as a readable markdown table,
    suitable for display in agent outputs.

    Args:
        df: pandas DataFrame to display
        title: Title for the display (default: "Data")

    Returns:
        Formatted string representation of the DataFrame
    """
    try:
        # Try to use ace_tools_open if available
        from ace_tools_open import display_dataframe_to_user as ace_display
        return ace_display(df, title)
    except ImportError:
        pass

    # Fallback: Native implementation
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return f"## {title}\n\nError: Expected a pandas DataFrame, got {type(df).__name__}"

        # Generate markdown table
        output_lines = [f"## {title}", ""]

        # Add shape info
        rows, cols = df.shape
        output_lines.append(f"*{rows} rows x {cols} columns*")
        output_lines.append("")

        # Convert to markdown
        if hasattr(df, 'to_markdown'):
            output_lines.append(df.to_markdown(index=True))
        else:
            # Manual markdown generation
            headers = [""] + list(df.columns)
            output_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            output_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for idx, row in df.iterrows():
                row_data = [str(idx)] + [str(v) for v in row.values]
                output_lines.append("| " + " | ".join(row_data) + " |")

        return "\n".join(output_lines)

    except Exception as e:
        return f"## {title}\n\nError displaying DataFrame: {e}"


@tool
def display_chart_to_user(
    data: Any,
    chart_type: str = "line",
    title: str = "Chart",
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
) -> str:
    """
    Display a chart visualization to the user.

    This tool creates a text-based description of chart data
    since direct image rendering is not available in text outputs.

    Args:
        data: Data to visualize (DataFrame, dict, or list)
        chart_type: Type of chart (line, bar, scatter, pie)
        title: Title for the chart
        x_column: Column name for x-axis (for DataFrames)
        y_column: Column name for y-axis (for DataFrames)

    Returns:
        Text description of the chart data
    """
    output_lines = [f"## {title} ({chart_type.capitalize()} Chart)", ""]

    try:
        import pandas as pd

        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return f"## {title}\n\nUnsupported data type: {type(data).__name__}"

        # Describe the data
        output_lines.append(f"**Data Shape:** {df.shape[0]} rows x {df.shape[1]} columns")
        output_lines.append("")

        if x_column and y_column and x_column in df.columns and y_column in df.columns:
            output_lines.append(f"**X-axis:** {x_column}")
            output_lines.append(f"**Y-axis:** {y_column}")
            output_lines.append("")

            # Summary statistics for y column
            y_stats = df[y_column].describe()
            output_lines.append(f"**{y_column} Statistics:**")
            output_lines.append(f"  - Min: {y_stats.get('min', 'N/A')}")
            output_lines.append(f"  - Max: {y_stats.get('max', 'N/A')}")
            output_lines.append(f"  - Mean: {y_stats.get('mean', 'N/A'):.2f}")
            output_lines.append(f"  - Std: {y_stats.get('std', 'N/A'):.2f}")
        else:
            # General description
            output_lines.append("**Columns:**")
            for col in df.columns:
                dtype = df[col].dtype
                output_lines.append(f"  - {col} ({dtype})")

        output_lines.append("")
        output_lines.append("*Note: Text-based output. For actual chart rendering, use a visualization library.*")

        return "\n".join(output_lines)

    except Exception as e:
        return f"## {title}\n\nError creating chart: {e}"


@tool
def display_json_to_user(data: Any, title: str = "JSON Data") -> str:
    """
    Display JSON data to the user in a formatted way.

    Args:
        data: Data to display (dict, list, or JSON-serializable object)
        title: Title for the display

    Returns:
        Formatted JSON string
    """
    import json

    output_lines = [f"## {title}", "", "```json"]

    try:
        if isinstance(data, str):
            # Try to parse if it's a JSON string
            data = json.loads(data)

        formatted = json.dumps(data, indent=2, default=str)
        output_lines.append(formatted)

    except Exception as e:
        output_lines.append(f"Error formatting JSON: {e}")
        output_lines.append(str(data))

    output_lines.append("```")
    return "\n".join(output_lines)


@tool
def display_summary_to_user(
    data: Any,
    title: str = "Summary",
    include_stats: bool = True,
) -> str:
    """
    Display a summary of data to the user.

    Provides statistical summary for numerical data and
    value counts for categorical data.

    Args:
        data: Data to summarize (DataFrame, Series, list, or dict)
        title: Title for the summary
        include_stats: Whether to include statistical information

    Returns:
        Formatted summary string
    """
    output_lines = [f"## {title}", ""]

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            output_lines.append(f"**Shape:** {data.shape[0]} rows x {data.shape[1]} columns")
            output_lines.append("")

            if include_stats:
                output_lines.append("### Numerical Columns")
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    stats_df = data[numeric_cols].describe()
                    output_lines.append(stats_df.to_markdown())
                else:
                    output_lines.append("*No numerical columns*")
                output_lines.append("")

                output_lines.append("### Categorical Columns")
                cat_cols = data.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    for col in cat_cols[:5]:  # Limit to first 5
                        output_lines.append(f"\n**{col}:**")
                        value_counts = data[col].value_counts().head(5)
                        for val, count in value_counts.items():
                            output_lines.append(f"  - {val}: {count}")
                else:
                    output_lines.append("*No categorical columns*")

        elif isinstance(data, pd.Series):
            output_lines.append(f"**Length:** {len(data)}")
            output_lines.append(f"**Type:** {data.dtype}")
            if include_stats and pd.api.types.is_numeric_dtype(data):
                output_lines.append("")
                output_lines.append(data.describe().to_string())
            else:
                output_lines.append("")
                output_lines.append("**Top Values:**")
                for val, count in data.value_counts().head(5).items():
                    output_lines.append(f"  - {val}: {count}")

        elif isinstance(data, (list, tuple)):
            output_lines.append(f"**Length:** {len(data)}")
            if len(data) > 0:
                output_lines.append(f"**First element type:** {type(data[0]).__name__}")
                if len(data) <= 10:
                    output_lines.append(f"**Values:** {data}")
                else:
                    output_lines.append(f"**First 5:** {data[:5]}")
                    output_lines.append(f"**Last 5:** {data[-5:]}")

        elif isinstance(data, dict):
            output_lines.append(f"**Keys:** {len(data)}")
            output_lines.append("")
            for key in list(data.keys())[:10]:
                val = data[key]
                val_preview = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                output_lines.append(f"  - {key}: {val_preview}")

        else:
            output_lines.append(f"**Type:** {type(data).__name__}")
            output_lines.append(f"**Value:** {str(data)[:500]}")

        return "\n".join(output_lines)

    except Exception as e:
        return f"## {title}\n\nError creating summary: {e}"


# List of all tools for easy import
ACE_TOOLS = [
    display_dataframe_to_user,
    display_chart_to_user,
    display_json_to_user,
    display_summary_to_user,
]


def get_ace_tools() -> List:
    """
    Get all ACE tools for adding to an agent.

    Returns:
        List of ACE tool functions
    """
    return ACE_TOOLS.copy()
