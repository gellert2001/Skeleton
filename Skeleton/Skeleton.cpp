//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Barcai Gellért Péter
// Neptun : TOE0OT
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
GPUProgram gpuProgram;

vec3 operator/(vec3 n1, vec3 n2)
{
	return vec3(n1.x / n2.x, n1.y / n2.y, n1.z / n2.z);
}
enum MaterialType { __Reflective__, __Rought__, __Refractive__ };
//elõadás dia alapján
struct Material
{
	vec3 ka, kd, ks;
	float shininess;
	float ior;
	vec3 F0;
	MaterialType materialType;
	Material(MaterialType __materialType) { this->materialType = __materialType; }
};
//elõadás dia alapján
struct RoughtMaterial :public Material
{
	RoughtMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(__Rought__)
	{
		this->ka = _kd * 3;
		this->kd = _kd;
		this->ks = _ks;
		this->shininess = _shininess;
	}
};
const vec3 one(1, 1, 1);
//elõadás dia alapján
struct ReflectiveMaterial :public Material
{
	ReflectiveMaterial(vec3 n, vec3 k) : Material(__Reflective__)
	{
		F0 = ((n - one) * (n - one) + k * k) / ((n + one) * (n + one) + k * k);
	}

};
//elõadás dia alapján

struct RefractiveMaterial : Material
{
	float n;

	RefractiveMaterial(vec3 n) : Material(__Refractive__)
	{
		F0 = ((n - one) * (n - one)) / ((n + one) * (n + one));
		ior = n.x;
	}
};
//elõadás dia alapján

struct Ray
{
	vec3 start;
	vec3 dir;
	bool out;
	Ray() {}
	Ray(vec3 __start, vec3 __dir)
	{
		this->start = __start;
		this->dir =normalize(__dir);
	}
};
//elõadás dia alapján

struct Hit
{
	float t;
	vec3 position;
	vec3 normal;
	Material* material;
	Hit() { t = -1; material = NULL; }

};
//elõadás dia alapján

class Camera {
public:
	vec3 eye, lookat, right, up;
	float fov;
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);

		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		float normalizedX = (2.0f * (X + 0.5f) / windowWidth - 1);
		float normalizedY = (2.0f * (Y + 0.5f) / windowHeight - 1);
		vec3 dir = lookat + right * normalizedX + up * normalizedY - eye;
		return Ray(eye, dir);
	}
};
//elõadás dia alapján

class Intersectable
{
protected:
	Material* material;
public:
	Intersectable() {}
	Intersectable(Material* __material__) :material(__material__) {}
	virtual Hit intersect(Ray& ray) = 0;
};

class Plane :public Intersectable
{
	vec3 n;
	vec3 P0;
public:
	Plane(vec3 __normalvec, vec3 __p0, Material* __material) : Intersectable(__material)
	{
		n = normalize(__normalvec);
		P0 = __p0;
	}
	Hit intersect(Ray& ray)
	{
		Hit ret_hit;
		ret_hit.t = dot((P0 - ray.start), n) / dot(ray.dir, n);
		vec3 p = ray.start + ray.dir * ret_hit.t;
		if ((p.x > 10 || p.x < -10 || p.z > 10 || p.z < -10))
			ret_hit.t = -1;

		Material* white_plane = new RoughtMaterial(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), 0);

		ret_hit.normal = n;
		ret_hit.position = ray.start + ray.dir * ret_hit.t;
		int compA = (int)abs(p.x) % 2;
		int compB = (int)abs(p.z) % 2;
		if (((p.x > 0 && p.z > 0) || (p.z < 0 && p.x < 0)) && (compA + compB) % 2 == 0)
			ret_hit.material = white_plane;
		if (((p.x < 0 && p.z > 0) || (p.z < 0 && p.x > 0)) && (compA + compB) % 2 == 1)
			ret_hit.material = white_plane;
		if (((p.x > 0 && p.z > 0) || (p.z < 0 && p.x < 0)) && (compA + compB) % 2 == 1)
			ret_hit.material = material;
		if (((p.x < 0 && p.z > 0) || (p.z < 0 && p.x > 0)) && (compA + compB) % 2 == 0)
			ret_hit.material = material;
		return ret_hit;
	}

};
//forrás: https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf?fbclid=IwAR0Q2wB1LSrgBc9VMo6Om6c605TQaB1BPBY84VLd4WLncxbYOVWUg0ZQkuU
class Cylinder : public Intersectable
{
	float radius;
	float height;
	vec3 v0;
	vec3 P0;
public:
	Cylinder(float __radius, float __height, vec3 __dirvec, vec3 __p0, Material* __material) : Intersectable(__material)
	{
		this->radius = __radius;
		this->height = __height;
		this->v0 = normalize(__dirvec);
		this->P0 = __p0;
	}

	

	Hit intersect(Ray& ray) override
	{
		Hit ret_hit;

		vec3 s = ray.start;
		vec3 d = ray.dir;

		vec3 deltap = s - P0;
		float a = dot(d - dot(d, v0) * v0, d - dot(d, v0) * v0);
		float b = 2 * dot(d - dot(d, v0) * v0, deltap - dot(deltap, v0) * v0);
		float c = dot(deltap - dot(deltap, v0) * v0, deltap - dot(deltap, v0) * v0) - radius * radius;

		float discr = b * b - 4 * a * c;

		if (discr < 0.0f)
			return ret_hit;

		float sqrtDiscr = sqrt(discr);
		float t1 = (-b + sqrtDiscr) / (2.0f * a);
		float t2 = (-b - sqrtDiscr) / (2.0f * a);

		float t = t1 < t2 ? t1 : t2;
		if (t < 0.0f)
			return ret_hit;

		vec3 P = s + t * d;
		vec3 deltaP = P - P0;
		float projection = dot(deltaP, (v0));
		if (projection < -1 || projection > height)
		{
			ret_hit.t = -1;
			return ret_hit;
		}

		ret_hit.t = t;
		ret_hit.material = this->material;
		ret_hit.position = P;
		ret_hit.normal = normalize((s + d * ret_hit.t) - P0 - v0 * dot((s + d * ret_hit.t) - P0, v0));
		return ret_hit;
	}
};

//forrás: https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf?fbclid=IwAR0Q2wB1LSrgBc9VMo6Om6c605TQaB1BPBY84VLd4WLncxbYOVWUg0ZQkuU
class Cone :public Intersectable
{
	vec3 Ptop;
	vec3 n;
	float angle;
	float height;

public:
	Cone(vec3 __top, vec3 __n, float __ang, float __height, Material* __material) : Intersectable(__material)
	{
		this->Ptop = __top;
		this->n = normalize(__n);
		this->angle = __ang;
		this->height = __height;
	}
	void gradf(Ray& ray, Hit& hit)
	{
		vec3 s = ray.start;
		vec3 d = ray.dir;
		float t = hit.t;
		if (t == -1) return;
		hit.normal = -1 * normalize(2 * dot(s + d * hit.t - Ptop, n) * n - 2 * (s + d * t - Ptop) * cosf(angle) * cosf(angle));
	}

	Hit intersect(Ray& ray) override
	{
		Hit ret_hit;
		vec3 d =ray.dir;
		vec3 s = ray.start;

		vec3 deltap = s - Ptop;
		float cosf2 = powf(cosf(angle), 2);
		float sinf2 = powf(sinf(angle), 2);

		float a = cosf2 * dot(d - dot(d, n) * n, d - dot(d, n) * n) - sinf2 * dot(d, n) * dot(d, n);
		float b = 2 * cosf2 * dot(d - dot(d, n) * n, deltap - dot(deltap, n) * n) - 2 * sinf2 * dot(d, n) * dot(deltap, n);
		float c = cosf2 * dot(deltap - dot(deltap, n) * n, deltap - dot(deltap, n) * n) - sinf2 * dot(deltap, n) * dot(deltap, n);

		float discr = powf(b, 2) - 4 * a * c;
		if (discr < 0)
			return ret_hit;
		else
		{
			float t1 = (-b + sqrt(discr)) / 2 / a;
			float t2 = (-b - sqrt(discr)) / 2 / a;
			ret_hit.t = t1 < t2 ? t1 : t2;
			ret_hit.material = this->material;
			gradf(ray, ret_hit);
			ret_hit.position = ray.start + ray.dir * ret_hit.t;
			float dis = length(ret_hit.position - Ptop);

			if (cosf(angle) * dis > 2 || ret_hit.position.y > Ptop.y)
				ret_hit.t = -1;

			return ret_hit;
		}
	}
};
const float epsilon = 0.001f;
//elõadás dia alapján
struct Light
{
	vec3 direction;
	vec3 Le;
	Light(vec3 __dir__, vec3 __le__)
	{
		this->direction = normalize(__dir__);
		this->Le = __le__;
	}
};
class Scene
{
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build()
	{
		lights.push_back(new Light(vec3(1.0f, 1.0f, 1.0f), vec3(2.0f, 2.0f, 2.0f)));
		La = vec3(0.4, 0.4, 0.4);
		camera.set(vec3(0, 1, 4), vec3(0, 0, 0), vec3(0, 1, 0), 45 * (M_PI / 180));
		Plane* myplane = new Plane(vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), new RoughtMaterial(vec3(0, 0.1, 0.3), vec3(0, 0, 0), 0));
		Cylinder* gold_cylinder = new Cylinder(
			0.3f,
			2,
			vec3(0.1f, 1.0f, 0.0f),
			vec3(1.0f, -1.0f, 0.0f),
			new ReflectiveMaterial(
				vec3(0.17f, 0.35f, 1.5f),
				vec3(3.1f, 2.7f, 1.9f)
			)
		);
		Cylinder* water_cylinder = new Cylinder(
			0.3f,
			2,
			vec3(-0.2f, 1.0f, -0.1f),
			vec3(0.0f, -1.0f, -0.8f),
			new RefractiveMaterial(vec3(1.3, 1.3, 1.3))
		);
		Cylinder* yellow_cylinder = new Cylinder(
			0.3f,
			2.0f,
			vec3(0.0f, 1.0f, 0.1f),
			vec3(-1.0f, -1.0f, 0.0f),
			new RoughtMaterial(
				vec3(0.3f, 0.2f, 0.1f),
				vec3(2.0f, 2.0f, 2.0f),
				50
			)
		);
		Cone* cyan_cone = new Cone(
			vec3(0.0f, 1.0f, 0.0f),
			vec3(-0.1f, -1.0f, -0.05f),
			0.2f,
			2,
			new RoughtMaterial(
				vec3(0.1f, 0.2f, 0.3f),
				vec3(2.0f, 2.0f, 2.0f),
				100
			)
		);
		Cone* magenta_cone = new Cone(
			vec3(0.0f, 1.0f, 0.8f),
			vec3(0.2, -1.0f, 0.0f),
			0.2f,
			2.0f,
			new RoughtMaterial(
				vec3(0.3f, 0.0f, 0.2f),
				vec3(2.0f, 2.0f, 2.0f),
				20
			)
		);
		objects.push_back(gold_cylinder);
		objects.push_back(water_cylinder);
		objects.push_back(myplane);
		objects.push_back(cyan_cone);
		objects.push_back(magenta_cone);
		objects.push_back(yellow_cylinder);
	}

	//elõadás dia alapján
	void render(std::vector<vec4>& image)
	{
		long timestart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++)
		{
#pragma omp parallel for 
			for (int X = 0; X < windowWidth; X++)
			{
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowHeight + X] = vec4(color.x, color.y, color.z, 1);

			}
		}
		printf("rendering time: %d millisencond", glutGet(GLUT_ELAPSED_TIME) - timestart);
	}
	//elõadás dia, és Szirmay-Kalos László youtube videója alapján
	vec3 trace(Ray& ray, int depth = 0)
	{
		if (depth > 5)
			return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0)
		{
			return La;
		}
		vec3 outRad(0, 0, 0);
		if (hit.material->materialType == __Rought__)
		{
			vec3 outRad = hit.material->ka * La;
			for (Light* l : lights)
			{
				Ray shadowRay(hit.position + normalize(hit.normal) * epsilon, l->direction);
				float cosTheta = dot(normalize(hit.normal), l->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay))
				{
					outRad = outRad + l->Le * hit.material->kd * cosTheta;
					vec3 halfWay = normalize(-ray.dir + l->direction);
					float cosDelta = dot((hit.normal), halfWay);
					if (cosDelta > 0)
					{
						outRad = outRad + l->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
					}
				}
			}
			return outRad;
		}
		else if (hit.material->materialType == __Reflective__)
		{
			float cosa = -dot(normalize(ray.dir), normalize(hit.normal));
			vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			outRad = outRad + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
			return outRad;

		}
		
		else if (hit.material->materialType == __Refractive__)
		{
			float ior = ray.out ? (hit.material->ior) : (1 / hit.material->ior);
			float cosa = -dot(ray.dir,hit.normal);
			float disc = 1 - (1 - cosa * cosa) / ior / ior;

			vec3 refractedDir = normalize(ray.dir / ior + hit.normal * (cosa / ior - sqrtf(disc)));
			if (length(refractedDir) > 0)
			{
				Ray refractedRay = Ray(hit.position - normalize(hit.normal) * epsilon, refractedDir);
				refractedRay.out = !ray.out;
				vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
				outRad =outRad +  trace(refractedRay, depth + 1) * (one - F);
			}
			return outRad;
		}
	}
	//elõadás dia alapján
	Hit firstIntersect(Ray& ray)
	{
		Hit bestHit;
		for (Intersectable* obj : objects)
		{
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
				bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)
			bestHit.normal = bestHit.normal;
		return bestHit;
	}
	//elõadás dia alapján
	bool shadowIntersect(Ray ray)
	{
		for (Intersectable* obj : objects)
		{
			if (obj->intersect(ray).t > 0)
				return true;
		}
		return false;
	}
	void rotate()
	{
		float angle = (45 * M_PI) /180;
		
		mat4 rotMatrix = RotationMatrix(angle, vec3(0.0f, 1.0f, 0.0f));
		vec4 __eye__(camera.eye.x, camera.eye.y, camera.eye.z, 1);
		vec4 __newEye__ = __eye__ * rotMatrix;
		camera.set(vec3(__newEye__.x, __newEye__.y, __newEye__.z), camera.lookat, vec3(0, 1, 0), camera.fov);
		
	}

};
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";
//Szirmay-Kalos László youtube videója alapján 
class FullScreenTexturedQuad {
	unsigned int vao, textureID;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW); 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	void loadTexture(std::vector<vec4>& image)
	{
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;

		if (location >= 0)
		{
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureID);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};
FullScreenTexturedQuad* fullScreenTexturedQuad;
Scene scene;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	scene.build();


	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);

	scene.render(image);

	fullScreenTexturedQuad->loadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a')
	{
		scene.rotate();
		glutPostRedisplay();
	}
}
void onKeyboardUp(unsigned char key, int pX, int pY) {
}
void onMouseMotion(int pX, int pY) {
}
void onMouse(int button, int state, int pX, int pY) {
}
void onIdle() {
}
