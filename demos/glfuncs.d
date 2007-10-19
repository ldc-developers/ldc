/*
 * Copyright (c) 2004-2006 Derelict Developers
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the names 'Derelict', 'DerelictGL', nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Converted to a static binding by Tomas Lindquist Olsen <tomas@famolsen.dk>
 */
module glfuncs;

import gltypes;

extern(System):

// Miscellaneous
void glClearIndex(GLfloat c);
void glClearColor(GLclampf,GLclampf,GLclampf,GLclampf);
void glClear(GLbitfield);
void glIndexMask(GLuint);
void glColorMask(GLboolean,GLboolean,GLboolean,GLboolean);
void glAlphaFunc(GLenum,GLclampf);
void glBlendFunc(GLenum,GLenum);
void glLogicOp(GLenum);
void glCullFace(GLenum);
void glFrontFace(GLenum);
void glPointSize(GLfloat);
void glLineWidth(GLfloat);
void glLineStipple(GLint,GLushort);
void glPolygonMode(GLenum,GLenum);
void glPolygonOffset(GLfloat,GLfloat);
void glPolygonStipple(GLubyte*);
void glGetPolygonStipple(GLubyte*);
void glEdgeFlag(GLboolean);
void glEdgeFlagv(GLboolean*);
void glScissor(GLint,GLint,GLsizei,GLsizei);
void glClipPlane(GLenum,GLdouble*);
void glGetClipPlane(GLenum,GLdouble*);
void glDrawBuffer(GLenum);
void glReadBuffer(GLenum);
void glEnable(GLenum);
void glDisable(GLenum);
GLboolean glIsEnabled(GLenum);
void glEnableClientState(GLenum);
void glDisableClientState(GLenum);
void glGetBooleanv(GLenum,GLboolean*);
void glGetDoublev(GLenum,GLdouble*);
void glGetFloatv(GLenum,GLfloat*);
void glGetIntegerv(GLenum,GLint*);
void glPushAttrib(GLbitfield);
void glPopAttrib();
void glPushClientAttrib(GLbitfield);
void glPopClientAttrib();
GLint glRenderMode(GLenum);
GLenum glGetError();
GLchar* glGetString(GLenum);
void glFinish();
void glFlush();
void glHint(GLenum,GLenum);

// Depth Buffer
void glClearDepth(GLclampd);
void glDepthFunc(GLenum);
void glDepthMask(GLboolean);
void glDepthRange(GLclampd,GLclampd);

// Accumulation Buffer
void glClearAccum(GLfloat,GLfloat,GLfloat,GLfloat);
void glAccum(GLenum,GLfloat);

// Transformation
void glMatrixMode(GLenum);
void glOrtho(GLdouble,GLdouble,GLdouble,GLdouble,GLdouble,GLdouble);
void glFrustum(GLdouble,GLdouble,GLdouble,GLdouble,GLdouble,GLdouble);
void glViewport(GLint,GLint,GLsizei,GLsizei);
void glPushMatrix();
void glPopMatrix();
void glLoadIdentity();
void glLoadMatrixd(GLdouble*);
void glLoadMatrixf(GLfloat*);
void glMultMatrixd(GLdouble*);
void glMultMatrixf(GLfloat*);
void glRotated(GLdouble,GLdouble,GLdouble,GLdouble);
void glRotatef(GLfloat,GLfloat,GLfloat,GLfloat);
void glScaled(GLdouble,GLdouble,GLdouble);
void glScalef(GLfloat,GLfloat,GLfloat);
void glTranslated(GLdouble,GLdouble,GLdouble);
void glTranslatef(GLfloat,GLfloat,GLfloat);

// Display Lists
GLboolean glIsList(GLuint);
void glDeleteLists(GLuint,GLsizei);
GLuint glGenLists(GLsizei);
void glNewList(GLuint,GLenum);
void glEndList();
void glCallList(GLuint);
void glCallLists(GLsizei,GLenum,GLvoid*);
void glListBase(GLuint);

// Drawing Functions
void glBegin(GLenum);
void glEnd();
void glVertex2d(GLdouble,GLdouble);
void glVertex2f(GLfloat,GLfloat);
void glVertex2i(GLint,GLint);
void glVertex2s(GLshort,GLshort);
void glVertex3d(GLdouble,GLdouble,GLdouble);
void glVertex3f(GLfloat,GLfloat,GLfloat);
void glVertex3i(GLint,GLint,GLint);
void glVertex3s(GLshort,GLshort,GLshort);
void glVertex4d(GLdouble,GLdouble,GLdouble,GLdouble);
void glVertex4f(GLfloat,GLfloat,GLfloat,GLfloat);
void glVertex4i(GLint,GLint,GLint,GLint);
void glVertex4s(GLshort,GLshort,GLshort,GLshort);
void glVertex2dv(GLdouble*);
void glVertex2fv(GLfloat*);
void glVertex2iv(GLint*);
void glVertex2sv(GLshort*);
void glVertex3dv(GLdouble*);
void glVertex3fv(GLfloat*);
void glVertex3iv(GLint*);
void glVertex3sv(GLshort*);
void glVertex4dv(GLdouble*);
void glVertex4fv(GLfloat*);
void glVertex4iv(GLint*);
void glVertex4sv(GLshort*);
void glNormal3b(GLbyte,GLbyte,GLbyte);
void glNormal3d(GLdouble,GLdouble,GLdouble);
void glNormal3f(GLfloat,GLfloat,GLfloat);
void glNormal3i(GLint,GLint,GLint);
void glNormal3s(GLshort,GLshort,GLshort);
void glNormal3bv(GLbyte*);
void glNormal3dv(GLdouble*);
void glNormal3fv(GLfloat*);
void glNormal3iv(GLint*);
void glNormal3sv(GLshort*);
void glIndexd(GLdouble);
void glIndexf(GLfloat);
void glIndexi(GLint);
void glIndexs(GLshort);
void glIndexub(GLubyte);
void glIndexdv(GLdouble*);
void glIndexfv(GLfloat*);
void glIndexiv(GLint*);
void glIndexsv(GLshort*);
void glIndexubv(GLubyte*);
void glColor3b(GLbyte,GLbyte,GLbyte);
void glColor3d(GLdouble,GLdouble,GLdouble);
void glColor3f(GLfloat,GLfloat,GLfloat);
void glColor3i(GLint,GLint,GLint);
void glColor3s(GLshort,GLshort,GLshort);
void glColor3ub(GLubyte,GLubyte,GLubyte);
void glColor3ui(GLuint,GLuint,GLuint);
void glColor3us(GLushort,GLushort,GLushort);
void glColor4b(GLbyte,GLbyte,GLbyte,GLbyte);
void glColor4d(GLdouble,GLdouble,GLdouble,GLdouble);
void glColor4f(GLfloat,GLfloat,GLfloat,GLfloat);
void glColor4i(GLint,GLint,GLint,GLint);
void glColor4s(GLshort,GLshort,GLshort,GLshort);
void glColor4ub(GLubyte,GLubyte,GLubyte,GLubyte);
void glColor4ui(GLuint,GLuint,GLuint,GLuint);
void glColor4us(GLushort,GLushort,GLushort,GLushort);
void glColor3bv(GLubyte*);
void glColor3dv(GLdouble*);
void glColor3fv(GLfloat*);
void glColor3iv(GLint*);
void glColor3sv(GLshort*);
void glColor3ubv(GLubyte*);
void glColor3uiv(GLuint*);
void glColor3usv(GLushort*);
void glColor4bv(GLbyte*);
void glColor4dv(GLdouble*);
void glColor4fv(GLfloat*);
void glColor4iv(GLint*);
void glColor4sv(GLshort*);
void glColor4ubv(GLubyte*);
void glColor4uiv(GLuint*);
void glColor4usv(GLushort);
void glTexCoord1d(GLdouble);
void glTexCoord1f(GLfloat);
void glTexCoord1i(GLint);
void glTexCoord1s(GLshort);
void glTexCoord2d(GLdouble,GLdouble);
void glTexCoord2f(GLfloat,GLfloat);
void glTexCoord2i(GLint,GLint);
void glTexCoord2s(GLshort,GLshort);
void glTexCoord3d(GLdouble,GLdouble,GLdouble);
void glTexCoord3f(GLfloat,GLfloat,GLfloat);
void glTexCoord3i(GLint,GLint,GLint);
void glTexCoord3s(GLshort,GLshort,GLshort);
void glTexCoord4d(GLdouble,GLdouble,GLdouble,GLdouble);
void glTexCoord4f(GLfloat,GLfloat,GLfloat,GLfloat);
void glTexCoord4i(GLint,GLint,GLint,GLint);
void glTexCoord4s(GLshort,GLshort,GLshort,GLshort);
void glTexCoord1dv(GLdouble*);
void glTexCoord1fv(GLfloat*);
void glTexCoord1iv(GLint*);
void glTexCoord1sv(GLshort*);
void glTexCoord2dv(GLdouble*);
void glTexCoord2fv(GLfloat*);
void glTexCoord2iv(GLint*);
void glTexCoord2sv(GLshort*);
void glTexCoord3dv(GLdouble*);
void glTexCoord3fv(GLfloat*);
void glTexCoord3iv(GLint*);
void glTexCoord3sv(GLshort*);
void glTexCoord4dv(GLdouble*);
void glTexCoord4fv(GLfloat*);
void glTexCoord4iv(GLint*);
void glTexCoord4sv(GLshort*);
void glRasterPos2d(GLdouble,GLdouble);
void glRasterPos2f(GLfloat,GLfloat);
void glRasterPos2i(GLint,GLint);
void glRasterPos2s(GLshort,GLshort);
void glRasterPos3d(GLdouble,GLdouble,GLdouble);
void glRasterPos3f(GLfloat,GLfloat,GLfloat);
void glRasterPos3i(GLint,GLint,GLint);
void glRasterPos3s(GLshort,GLshort,GLshort);
void glRasterPos4d(GLdouble,GLdouble,GLdouble,GLdouble);
void glRasterPos4f(GLfloat,GLfloat,GLfloat,GLfloat);
void glRasterPos4i(GLint,GLint,GLint,GLint);
void glRasterPos4s(GLshort,GLshort,GLshort,GLshort);
void glRasterPos2dv(GLdouble*);
void glRasterPos2fv(GLfloat*);
void glRasterPos2iv(GLint*);
void glRasterPos2sv(GLshort*);
void glRasterPos3dv(GLdouble*);
void glRasterPos3fv(GLfloat*);
void glRasterPos3iv(GLint*);
void glRasterPos3sv(GLshort*);
void glRasterPos4dv(GLdouble*);
void glRasterPos4fv(GLfloat*);
void glRasterPos4iv(GLint*);
void glRasterPos4sv(GLshort*);
void glRectd(GLdouble,GLdouble,GLdouble,GLdouble);
void glRectf(GLfloat,GLfloat,GLfloat,GLfloat);
void glRecti(GLint,GLint,GLint,GLint);
void glRects(GLshort,GLshort,GLshort,GLshort);
void glRectdv(GLdouble*);
void glRectfv(GLfloat*);
void glRectiv(GLint*);
void glRectsv(GLshort*);

// Lighting
void glShadeModel(GLenum);
void glLightf(GLenum,GLenum,GLfloat);
void glLighti(GLenum,GLenum,GLint);
void glLightfv(GLenum,GLenum,GLfloat*);
void glLightiv(GLenum,GLenum,GLint*);
void glGetLightfv(GLenum,GLenum,GLfloat*);
void glGetLightiv(GLenum,GLenum,GLint*);
void glLightModelf(GLenum,GLfloat);
void glLightModeli(GLenum,GLint);
void glLightModelfv(GLenum,GLfloat*);
void glLightModeliv(GLenum,GLint*);
void glMaterialf(GLenum,GLenum,GLfloat);
void glMateriali(GLenum,GLenum,GLint);
void glMaterialfv(GLenum,GLenum,GLfloat*);
void glMaterialiv(GLenum,GLenum,GLint*);
void glGetMaterialfv(GLenum,GLenum,GLfloat*);
void glGetMaterialiv(GLenum,GLenum,GLint*);
void glColorMaterial(GLenum,GLenum);

// Raster functions
void glPixelZoom(GLfloat,GLfloat);
void glPixelStoref(GLenum,GLfloat);
void glPixelStorei(GLenum,GLint);
void glPixelTransferf(GLenum,GLfloat);
void glPixelTransferi(GLenum,GLint);
void glPixelMapfv(GLenum,GLint,GLfloat*);
void glPixelMapuiv(GLenum,GLint,GLuint*);
void glPixelMapusv(GLenum,GLint,GLushort*);
void glGetPixelMapfv(GLenum,GLfloat*);
void glGetPixelMapuiv(GLenum,GLuint*);
void glGetPixelMapusv(GLenum,GLushort*);
void glBitmap(GLsizei,GLsizei,GLfloat,GLfloat,GLfloat,GLfloat,GLubyte*);
void glReadPixels(GLint,GLint,GLsizei,GLsizei,GLenum,GLenum,GLvoid*);
void glDrawPixels(GLsizei,GLsizei,GLenum,GLenum,GLvoid*);
void glCopyPixels(GLint,GLint,GLsizei,GLsizei,GLenum);

// Stenciling
void glStencilFunc(GLenum,GLint,GLuint);
void glStencilMask(GLuint);
void glStencilOp(GLenum,GLenum,GLenum);
void glClearStencil(GLint);

// Texture mapping
void glTexGend(GLenum,GLenum,GLdouble);
void glTexGenf(GLenum,GLenum,GLfloat);
void glTexGeni(GLenum,GLenum,GLint);
void glTexGendv(GLenum,GLenum,GLdouble*);
void glTexGenfv(GLenum,GLenum,GLfloat*);
void glTexGeniv(GLenum,GLenum,GLint*);
void glGetTexGendv(GLenum,GLenum,GLdouble*);
void glGetTexGenfv(GLenum,GLenum,GLfloat*);
void glGetTexGeniv(GLenum,GLenum,GLint*);
void glTexEnvf(GLenum,GLenum,GLfloat);
void glTexEnvi(GLenum,GLenum,GLint);
void glTexEnvfv(GLenum,GLenum,GLfloat*);
void glTexEnviv(GLenum,GLenum,GLint*);
void glGetTexEnvfv(GLenum,GLenum,GLfloat*);
void glGetTexEnviv(GLenum,GLenum,GLint*);
void glTexParameterf(GLenum,GLenum,GLfloat);
void glTexParameteri(GLenum,GLenum,GLint);
void glTexParameterfv(GLenum,GLenum,GLfloat*);
void glTexParameteriv(GLenum,GLenum,GLint*);
void glGetTexParameterfv(GLenum,GLenum,GLfloat*);
void glGetTexParameteriv(GLenum,GLenum,GLint*);
void glGetTexLevelParameterfv(GLenum,GLint,GLenum,GLfloat*);
void glGetTexLevelParameteriv(GLenum,GLint,GLenum,GLint*);
void glTexImage1D(GLenum,GLint,GLint,GLsizei,GLint,GLenum,GLenum,GLvoid*);
void glTexImage2D(GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,GLvoid*);
void glGetTexImage(GLenum,GLint,GLenum,GLenum,GLvoid*);

// Evaluators
void glMap1d(GLenum,GLdouble,GLdouble,GLint,GLint,GLdouble*);
void glMap1f(GLenum,GLfloat,GLfloat,GLint,GLint,GLfloat*);
void glMap2d(GLenum,GLdouble,GLdouble,GLint,GLint,GLdouble,GLdouble,GLint,GLint,GLdouble*);
void glMap2f(GLenum,GLfloat,GLfloat,GLint,GLint,GLfloat,GLfloat,GLint,GLint,GLfloat*);
void glGetMapdv(GLenum,GLenum,GLdouble*);
void glGetMapfv(GLenum,GLenum,GLfloat*);
void glGetMapiv(GLenum,GLenum,GLint*);
void glEvalCoord1d(GLdouble);
void glEvalCoord1f(GLfloat);
void glEvalCoord1dv(GLdouble*);
void glEvalCoord1fv(GLfloat*);
void glEvalCoord2d(GLdouble,GLdouble);
void glEvalCoord2f(GLfloat,GLfloat);
void glEvalCoord2dv(GLdouble*);
void glEvalCoord2fv(GLfloat*);
void glMapGrid1d(GLint,GLdouble,GLdouble);
void glMapGrid1f(GLint,GLfloat,GLfloat);
void glMapGrid2d(GLint,GLdouble,GLdouble,GLint,GLdouble,GLdouble);
void glMapGrid2f(GLint,GLfloat,GLfloat,GLint,GLfloat,GLfloat);
void glEvalPoint1(GLint);
void glEvalPoint2(GLint,GLint);
void glEvalMesh1(GLenum,GLint,GLint);
void glEvalMesh2(GLenum,GLint,GLint,GLint,GLint);

// Fog
void glFogf(GLenum,GLfloat);
void glFogi(GLenum,GLint);
void glFogfv(GLenum,GLfloat*);
void glFogiv(GLenum,GLint*);

// Selection and Feedback
void glFeedbackBuffer(GLsizei,GLenum,GLfloat*);
void glPassThrough(GLfloat);
void glSelectBuffer(GLsizei,GLuint*);
void glInitNames();
void glLoadName(GLuint);
void glPushName(GLuint);
void glPopName();

// texture objects
void glGenTextures(GLsizei,GLuint*);
void glDeleteTextures(GLsizei,GLuint*);
void glBindTexture(GLenum,GLuint);
void glPrioritizeTextures(GLsizei,GLuint*,GLclampf*);
GLboolean glAreTexturesResident(GLsizei,GLuint*,GLboolean*);
GLboolean glIsTexture(GLuint);

// texture mapping
void glTexSubImage1D(GLenum,GLint,GLint,GLsizei,GLenum,GLenum,GLvoid*);
void glTexSubImage2D(GLenum,GLint,GLint,GLint,GLsizei,GLsizei,GLenum,GLenum,GLvoid*);
void glCopyTexImage1D(GLenum,GLint,GLenum,GLint,GLint,GLsizei,GLint);
void glCopyTexImage2D(GLenum,GLint,GLenum,GLint,GLint,GLsizei,GLsizei,GLint);
void glCopyTexSubImage1D(GLenum,GLint,GLint,GLint,GLint,GLsizei);
void glCopyTexSubImage2D(GLenum,GLint,GLint,GLint,GLint,GLint,GLsizei,GLsizei);

// vertex arrays
void glVertexPointer(GLint,GLenum,GLsizei,GLvoid*);
void glNormalPointer(GLenum,GLsizei,GLvoid*);
void glColorPointer(GLint,GLenum,GLsizei,GLvoid*);
void glIndexPointer(GLenum,GLsizei,GLvoid*);
void glTexCoordPointer(GLint,GLenum,GLsizei,GLvoid*);
void glEdgeFlagPointer(GLsizei,GLvoid*);
void glGetPointerv(GLenum,GLvoid**);
void glArrayElement(GLint);
void glDrawArrays(GLenum,GLint,GLsizei);
void glDrawElements(GLenum,GLsizei,GLenum,GLvoid*);
void glInterleavedArrays(GLenum,GLsizei,GLvoid*);
