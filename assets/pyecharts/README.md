# 離線圖表資產

模擬器在產生軌跡 HTML 時使用這兩份 JavaScript。兩份檔案均與下表的上游發行檔逐位元相同。

| 檔案 | 版本與來源 | SHA-256 |
| --- | --- | --- |
| `echarts.min.js` | [Apache ECharts 6.0.0](https://github.com/apache/echarts/blob/6.0.0/dist/echarts.min.js) | `baa8dfe7e1d9336b98e8986ba7e20ea15e7cdbea1ef42a59d59478632fa45a1d` |
| `echarts-gl.min.js` | [ECharts GL 2.0.8](https://github.com/ecomfe/echarts-gl/blob/2.0.8/dist/echarts-gl.min.js) | `1c757f1cf9ba6d23237ded823c73e42ebfb0dafa3aff636554a753abb1829a6f` |

## 授權與聲明

- Apache ECharts 採 Apache-2.0。保留 [LICENSE](licenses/ECharts.LICENSE.txt)、[NOTICE](licenses/ECharts.NOTICE.txt) 及內含的 [d3 BSD-3-Clause 授權](licenses/d3.LICENSE.txt)。[上游發行標籤](https://github.com/apache/echarts/tree/6.0.0)
- ECharts GL 採 BSD-3-Clause。保留 [LICENSE](licenses/ECharts-GL.LICENSE.txt)、[發行檔版權聲明](licenses/ECharts-GL.dist.NOTICE.txt) 及 [隨壓縮檔發布的 Microsoft 聲明](echarts-gl.min.js.LICENSE.txt)。其內含的 ECharts 5.1.2 程式碼共用上列 ECharts 與 d3 授權文本，另附 [5.1.2 的 NOTICE](licenses/ECharts-5.1.2.NOTICE.txt)。[ECharts GL 上游發行標籤](https://github.com/ecomfe/echarts-gl/tree/2.0.8) · [ECharts 5.1.2 發行包](https://registry.npmjs.org/echarts/-/echarts-5.1.2.tgz)
- 上表兩份資產分別打包 zrender 6.0.0 與 5.1.1，兩個版本共用同一份 [BSD-3-Clause 授權文本](licenses/ZRender.LICENSE.txt)。[zrender 6.0.0 上游授權](https://github.com/ecomfe/zrender/blob/6.0.0/LICENSE) · [zrender 5.1.1 上游授權](https://github.com/ecomfe/zrender/blob/5.1.1/LICENSE)
- ECharts GL 另含 ClayGL 1.3.0、ClayGL 內的 gl-matrix 及改編自 SunCalc 的程式碼。保留 [ClayGL 授權](licenses/ClayGL.LICENSE.txt)、[gl-matrix 授權](licenses/gl-matrix.LICENSE.txt) 及 [SunCalc 授權](licenses/SunCalc.LICENSE.txt)。[ClayGL 上游授權](https://github.com/pissang/claygl/blob/1.3.0/LICENSE) · [gl-matrix 聲明來源](https://github.com/pissang/claygl/blob/1.3.0/src/glmatrix/vec2.js) · [SunCalc 署名來源](https://github.com/ecomfe/echarts-gl/blob/2.0.8/src/util/sunCalc.js) · [SunCalc 上游授權](https://github.com/mourner/suncalc/blob/v1.8.0/LICENSE)
